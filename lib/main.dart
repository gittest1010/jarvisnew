import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle, ByteData;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart';

// --- UTILITY ---
Float32List convertBytesToFloat32(Uint8List bytes) {
  final int length = bytes.length ~/ 2;
  final Float32List float32List = Float32List(length);
  final ByteData byteData = ByteData.sublistView(bytes);

  for (int i = 0; i < length; i++) {
    int sample = byteData.getInt16(i * 2, Endian.little);
    float32List[i] = sample / 32768.0;
  }
  return float32List;
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(
      debugShowCheckedModeBanner: false, home: VoiceAssistantScreen()));
}

class VoiceAssistantScreen extends StatefulWidget {
  const VoiceAssistantScreen({super.key});

  @override
  State<VoiceAssistantScreen> createState() => _VoiceAssistantScreenState();
}

class _VoiceAssistantScreenState extends State<VoiceAssistantScreen> {
  final TextEditingController _controller = TextEditingController();
  String _statusText = "Starting...";
  bool _isLoading = true;

  late final AudioRecorder _audioRecorder;
  late final AudioPlayer _audioPlayer;

  sherpa_onnx.OnlineRecognizer? _sttRecognizer;
  sherpa_onnx.OnlineStream? _sttStream;
  sherpa_onnx.OfflineTts? _ttsEngine;

  bool _isRecording = false;
  String _lastRecognizedText = '';
  int _sentenceIndex = 0;
  final int _sampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _audioRecorder = AudioRecorder();
    _audioPlayer = AudioPlayer();
    _startSetupProcess();
  }

  Future<void> _startSetupProcess() async {
    // 1. Permissions
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() {
        _statusText = "Mic Permission Denied ❌";
        _isLoading = false;
      });
      return;
    }
    // 2. Init
    await _initSherpaEngines();
  }

  // --- ASSETS COPYING ---
  Future<String> _copyAssetToFile(String assetPath) async {
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final fileName = assetPath.split('/').last;
      final file = File('${docsDir.path}/$fileName');

      // Check existence
      if (!await file.exists()) {
        setState(() => _statusText = "Copying $fileName...");
        final data = await rootBundle.load(assetPath);
        await file.writeAsBytes(data.buffer.asUint8List(), flush: true);
      }
      return file.path;
    } catch (e) {
      throw Exception("Copy Failed: $assetPath");
    }
  }

  Future<String> _copyAssetFolder(String assetFolderPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final localFolder =
        Directory('${docsDir.path}/${assetFolderPath.split('/').last}');

    // Logic Change: Agar folder hai par khali hai, to wapas copy karo
    bool exists = await localFolder.exists();
    if (exists) {
      if (await localFolder.list().isEmpty) {
        exists = false; // Re-copy needed
      }
    }

    if (!exists) {
      await localFolder.create(recursive: true);
    } else {
      return localFolder.path; // Already ready
    }

    try {
      final manifestContent = await rootBundle.loadString('AssetManifest.json');
      final Map<String, dynamic> manifestMap = json.decode(manifestContent);

      final filePaths = manifestMap.keys
          .where((String key) => key.contains('espeak-ng-data'))
          .toList();

      if (filePaths.isEmpty) {
        throw Exception("espeak data not found in Assets!");
      }

      for (final filePath in filePaths) {
        final relativePathIndex = filePath.indexOf('espeak-ng-data/');
        if (relativePathIndex == -1) continue;

        final relativePath = filePath.substring(relativePathIndex);
        final localFilePath = '${docsDir.path}/$relativePath';
        final localFile = File(localFilePath);

        if (!await localFile.exists()) {
          setState(() => _statusText = "Unpacking Data...");
          await localFile.parent.create(recursive: true);
          final data = await rootBundle.load(filePath);
          await localFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
        }
      }
    } catch (e) {
      print("Folder Copy Error: $e");
    }
    return localFolder.path;
  }

  // --- INIT SHERPA ---
  Future<void> _initSherpaEngines() async {
    try {
      setState(() => _statusText = "Initializing AI Models...");

      // 1. Copy Files
      final encoderPath =
          await _copyAssetToFile('assets/tiny-encoder.int8.onnx');
      final decoderPath =
          await _copyAssetToFile('assets/tiny-decoder.int8.onnx');
      final tokensPath = await _copyAssetToFile('assets/tokens.txt');

      final ttsModelPath =
          await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx');
      final ttsJsonPath =
          await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx.json');

      // 2. Copy Folder
      final espeakDataPath = await _copyAssetFolder('assets/espeak-ng-data');

      // 3. Configure STT
      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: encoderPath,
            decoder: decoderPath,
            joiner: tokensPath,
          ),
          tokens: tokensPath,
        ),
      );
      _sttRecognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      // 4. Configure TTS
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: ttsModelPath,
            tokens: ttsJsonPath,
            dataDir: espeakDataPath,
          ),
          provider: 'sherpa-onnx',
        ),
      );
      _ttsEngine = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() {
        _isLoading = false;
        _statusText = "Jarvis Ready. Tap Mic 🎙️";
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = "Init Error: $e";
      });
      print(e);
    }
  }

  // --- RECORDING ---
  Future<void> _toggleRecording() async {
    if (_sttRecognizer == null || _isLoading) return;

    if (_isRecording) {
      await _audioRecorder.stop();
      setState(() => _isRecording = false);
    } else {
      _sttStream?.free();
      _sttStream = _sttRecognizer?.createStream();

      if (await _audioRecorder.hasPermission()) {
        setState(() => _isRecording = true);

        final stream = await _audioRecorder.startStream(const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
        ));

        stream.listen((data) {
          final samples = convertBytesToFloat32(Uint8List.fromList(data));
          _sttStream!.acceptWaveform(samples: samples, sampleRate: _sampleRate);

          while (_sttRecognizer!.isReady(_sttStream!)) {
            _sttRecognizer!.decode(_sttStream!);
          }

          final text = _sttRecognizer!.getResult(_sttStream!).text;
          bool isEndpoint = _sttRecognizer!.isEndpoint(_sttStream!);
          _updateUI(text, isEndpoint);

          if (isEndpoint) _sttRecognizer!.reset(_sttStream!);
        });
      }
    }
  }

  void _updateUI(String text, bool isEndpoint) {
    if (text.isNotEmpty) {
      String fullText = '$_sentenceIndex: $text\n$_lastRecognizedText';
      _controller.value = TextEditingValue(
        text: fullText,
        selection: TextSelection.collapsed(offset: fullText.length),
      );

      if (isEndpoint) {
        _lastRecognizedText = fullText;
        _sentenceIndex++;
        _toggleRecording().then((_) => _speak(text));
      }
    }
  }

  // --- SPEAKING ---
  Future<void> _speak(String text) async {
    if (_ttsEngine == null) return;

    final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
    if (audio.samples.isEmpty) return;

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/tts_output.wav');
    await file.writeAsBytes(_createWavHeader(audio.samples, audio.sampleRate),
        flush: true);
    await _audioPlayer.play(DeviceFileSource(file.path));
  }

  Uint8List _createWavHeader(Float32List samples, int sampleRate) {
    int numChannels = 1;
    int bitsPerSample = 16;
    int byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
    int blockAlign = numChannels * bitsPerSample ~/ 8;
    int dataSize = samples.length * numChannels * bitsPerSample ~/ 8;
    int chunkSize = 36 + dataSize;

    var buffer = BytesBuilder();
    buffer.add('RIFF'.codeUnits);
    buffer.add(_int32(chunkSize));
    buffer.add('WAVE'.codeUnits);
    buffer.add('fmt '.codeUnits);
    buffer.add(_int32(16));
    buffer.add(_int16(1));
    buffer.add(_int16(numChannels));
    buffer.add(_int32(sampleRate));
    buffer.add(_int32(byteRate));
    buffer.add(_int16(blockAlign));
    buffer.add(_int16(bitsPerSample));
    buffer.add('data'.codeUnits);
    buffer.add(_int32(dataSize));

    for (var sample in samples) {
      double s = sample * 1.5;
      if (s > 1.0) s = 1.0;
      if (s < -1.0) s = -1.0;
      buffer.add(_int16((s * 32767).toInt()));
    }
    return buffer.toBytes();
  }

  List<int> _int32(int v) =>
      Uint8List(4)..buffer.asByteData().setInt32(0, v, Endian.little);
  List<int> _int16(int v) =>
      Uint8List(2)..buffer.asByteData().setInt16(0, v, Endian.little);

  @override
  void dispose() {
    _audioRecorder.dispose();
    _audioPlayer.dispose();
    _sttStream?.free();
    _sttRecognizer?.free();
    _ttsEngine?.free();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
          title: const Text('Jarvis AI'), backgroundColor: Colors.blueAccent),
      body: Stack(
        children: [
          // 1. Main Content (Behind the overlay)
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                // Simplified Status Bar
                Container(
                  padding: const EdgeInsets.all(15),
                  decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(10)),
                  child: Row(
                    children: [
                      const Icon(Icons.info, color: Colors.blueAccent),
                      const SizedBox(width: 10),
                      Expanded(
                          child: Text(_statusText,
                              style: const TextStyle(
                                  fontWeight: FontWeight.bold))),
                    ],
                  ),
                ),
                const SizedBox(height: 20),

                // Text Area
                Expanded(
                  child: TextField(
                    controller: _controller,
                    maxLines: null,
                    readOnly: true,
                    decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        hintText: "Wait for initialization..."),
                  ),
                ),
                const SizedBox(height: 30),

                // Button
                GestureDetector(
                  onTap: _isLoading ? null : _toggleRecording,
                  child: CircleAvatar(
                    radius: 40,
                    backgroundColor: _isLoading
                        ? Colors.grey
                        : (_isRecording ? Colors.red : Colors.blue),
                    child: Icon(_isRecording ? Icons.stop : Icons.mic,
                        color: Colors.white, size: 40),
                  ),
                ),
              ],
            ),
          ),

          // 2. Full Screen Loading Overlay (Prevents Touch)
          if (_isLoading)
            Container(
              height: double.infinity,
              width: double.infinity,
              color: Colors.black
                  .withOpacity(0.7), // Semi-transparent black background
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(
                    color: Colors.white,
                    strokeWidth: 4,
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    "Setting up Jarvis...",
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 10),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 40),
                    child: Text(
                      _statusText, // Shows 'Unpacking data...', 'Copying...' etc.
                      textAlign: TextAlign.center,
                      style:
                          const TextStyle(color: Colors.white70, fontSize: 16),
                    ),
                  ),
                  const SizedBox(height: 30),
                  const Text(
                    "(This happens only once)",
                    style: TextStyle(
                        color: Colors.white54,
                        fontSize: 14,
                        fontStyle: FontStyle.italic),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
