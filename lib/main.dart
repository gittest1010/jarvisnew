import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle, ByteData;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart';
// Archive package for extracting .tar.bz2
import 'package:archive/archive.dart';
import 'package:archive/archive_io.dart';

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

  // STT: Streaming Recognizer (Runs locally)
  sherpa_onnx.OnlineRecognizer? _sttRecognizer;
  sherpa_onnx.OnlineStream? _sttStream;

  // TTS: Offline Engine (Runs locally)
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
    // 1. Permissions check
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() {
        _statusText = "Mic Permission Denied ❌";
        _isLoading = false;
      });
      return;
    }
    // 2. Start Initialization
    await _initSherpaEngines();
  }

  // --- ASSETS COPYING LOGIC ---
  Future<String> _copyAssetToFile(String assetPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${docsDir.path}/$fileName');

    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List(), flush: true);
    }
    return file.path;
  }

  // --- EXTRACT TAR.BZ2 (For Offline TTS Data) ---
  Future<String> _extractEspeakData() async {
    final docsDir = await getApplicationDocumentsDirectory();
    final dataDir = Directory('${docsDir.path}/espeak-ng-data');

    if (await dataDir.exists()) {
      return dataDir.path;
    }

    setState(() => _statusText = "Extracting Offline TTS Data...");

    final data = await rootBundle.load('assets/espeak-ng-data.tar.bz2');
    final bytes = data.buffer.asUint8List();

    final archive = TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(bytes));

    for (final file in archive) {
      final filename = file.name;
      if (file.isFile) {
        final data = file.content as List<int>;
        File('${docsDir.path}/$filename')
          ..createSync(recursive: true)
          ..writeAsBytesSync(data);
      }
    }

    return dataDir.path;
  }

  // --- INIT SHERPA (OFFLINE SETUP) ---
  Future<void> _initSherpaEngines() async {
    try {
      setState(() => _statusText = "Initializing Native Libs...");

      // STEP 1: Load C++ Libraries
      sherpa_onnx.initBindings();

      setState(() => _statusText = "Copying Models...");

      // STEP 2: Copy Assets to Internal Storage
      final encoderPath =
          await _copyAssetToFile('assets/tiny-encoder.int8.onnx');
      final decoderPath =
          await _copyAssetToFile('assets/tiny-decoder.int8.onnx');
      final tokensPath = await _copyAssetToFile('assets/tokens.txt');

      final ttsModelPath =
          await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx');
      final ttsJsonPath =
          await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx.json');

      final espeakDataPath = await _extractEspeakData();

      setState(() => _statusText = "Configuring AI...");

      // STEP 3: STT (Speech to Text) Setup - Streaming but Local
      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: encoderPath,
            decoder: decoderPath,
            joiner: tokensPath,
          ),
          tokens: tokensPath,
          numThreads: 1,
        ),
      );
      _sttRecognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      // STEP 4: TTS (Text to Speech) Setup - STRICTLY OFFLINE
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: ttsModelPath, // Local ONNX model
            tokens: ttsJsonPath, // Local tokens
            dataDir: espeakDataPath, // Local espeak-ng-data
          ),
          provider: 'sherpa-onnx',
          numThreads: 1,
        ),
      );
      _ttsEngine = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() {
        _isLoading = false;
        _statusText = "Jarvis Ready (Offline). Tap Mic 🎙️";
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = "Error: $e";
      });
      print("INIT ERROR: $e");
    }
  }

  // --- RECORDING & PROCESSING ---
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
          encoder: AudioEncoder.wav, // Ensure WAV format for compatibility
          sampleRate: 16000,
          numChannels: 1,
        ));

        stream.listen((data) {
          final samples = convertBytesToFloat32(Uint8List.fromList(data));

          if (_sttStream != null) {
            _sttStream!
                .acceptWaveform(samples: samples, sampleRate: _sampleRate);

            while (_sttRecognizer!.isReady(_sttStream!)) {
              _sttRecognizer!.decode(_sttStream!);
            }

            final text = _sttRecognizer!.getResult(_sttStream!).text;
            bool isEndpoint = _sttRecognizer!.isEndpoint(_sttStream!);
            _updateUI(text, isEndpoint);

            if (isEndpoint) {
              _sttRecognizer!.reset(_sttStream!);
            }
          }
        });
      }
    }
  }

  void _updateUI(String text, bool isEndpoint) {
    if (text.isNotEmpty) {
      String fullText = '$_sentenceIndex: $text\n$_lastRecognizedText';

      if (!mounted) return;

      setState(() {
        _controller.value = TextEditingValue(
          text: fullText,
          selection: TextSelection.collapsed(offset: fullText.length),
        );
      });

      if (isEndpoint) {
        _lastRecognizedText = fullText;
        _sentenceIndex++;
        _speak(text);
      }
    }
  }

  // --- SPEAKING (OFFLINE TTS) ---
  Future<void> _speak(String text) async {
    if (_ttsEngine == null) return;

    // Generates audio locally on device
    final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
    if (audio.samples.isEmpty) return;

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/tts_output.wav');

    await file.writeAsBytes(_createWavHeader(audio.samples, audio.sampleRate),
        flush: true);

    await _audioPlayer.play(DeviceFileSource(file.path));
  }

  // WAV Header Generator
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
      double s = sample;
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
          title: const Text('Jarvis AI (Offline)'),
          backgroundColor: Colors.blueAccent),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
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
                Expanded(
                  child: TextField(
                    controller: _controller,
                    maxLines: null,
                    readOnly: true,
                    decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        hintText: "Transcription will appear here..."),
                  ),
                ),
                const SizedBox(height: 30),
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
          if (_isLoading)
            Container(
              height: double.infinity,
              width: double.infinity,
              color: Colors.black.withOpacity(0.7),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(color: Colors.white),
                  const SizedBox(height: 20),
                  const Text("Setting up Jarvis...",
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  Text(_statusText,
                      style: const TextStyle(color: Colors.white70)),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
