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
  String _statusText = "Initializing...";

  late final AudioRecorder _audioRecorder;
  late final AudioPlayer _audioPlayer;

  sherpa_onnx.OnlineRecognizer? _sttRecognizer;
  sherpa_onnx.OnlineStream? _sttStream;
  sherpa_onnx.OfflineTts? _ttsEngine;

  bool _isInitialized = false;
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
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() => _statusText = "Permission Denied ❌");
      return;
    }
    await _initSherpaEngines();
  }

  // --- ASSETS ---
  Future<String> _copyAssetToFile(String assetPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${docsDir.path}/$fileName');
    if (!await file.exists()) {
      setState(() => _statusText = "Copying $fileName...");
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List(), flush: true);
    }
    return file.path;
  }

  Future<String> _copyAssetFolder(String assetFolderPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final localFolder =
        Directory('${docsDir.path}/${assetFolderPath.split('/').last}');
    if (await localFolder.exists()) return localFolder.path;

    await localFolder.create(recursive: true);
    try {
      final manifestContent = await rootBundle.loadString('AssetManifest.json');
      final Map<String, dynamic> manifestMap = json.decode(manifestContent);
      final filePaths =
          manifestMap.keys.where((k) => k.contains('espeak-ng-data')).toList();

      for (final filePath in filePaths) {
        final relativePathIndex = filePath.indexOf('espeak-ng-data/');
        if (relativePathIndex == -1) continue;
        final relativePath = filePath.substring(relativePathIndex);
        final localFile = File('${docsDir.path}/$relativePath');
        if (!await localFile.exists()) {
          setState(() => _statusText = "Unpacking data...");
          await localFile.parent.create(recursive: true);
          final data = await rootBundle.load(filePath);
          await localFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
        }
      }
    } catch (e) {
      print(e);
    }
    return localFolder.path;
  }

  // --- INIT ---
  Future<void> _initSherpaEngines() async {
    try {
      setState(() => _statusText = "Loading AI...");
      final encoder = await _copyAssetToFile('assets/tiny-encoder.int8.onnx');
      final decoder = await _copyAssetToFile('assets/tiny-decoder.int8.onnx');
      final tokens = await _copyAssetToFile('assets/tokens.txt');
      final ttsModel =
          await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx');
      final ttsJson =
          await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx.json');
      final espeakData = await _copyAssetFolder('assets/espeak-ng-data');

      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: encoder,
            decoder: decoder,
            joiner: tokens,
          ),
          tokens: tokens,
        ),
      );
      _sttRecognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: ttsModel,
            tokens: ttsJson,
            dataDir: espeakData,
          ),
          provider: 'sherpa-onnx',
        ),
      );
      _ttsEngine = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() {
        _isInitialized = true;
        _statusText = "Ready. Tap Mic 🎙️";
      });
    } catch (e) {
      setState(() => _statusText = "Error: $e");
    }
  }

  // --- RECORD & SPEAK ---
  Future<void> _toggleRecording() async {
    if (!_isInitialized) return;
    if (_isRecording) {
      await _audioRecorder.stop();
      setState(() => _isRecording = false);
    } else {
      _sttStream?.free();
      _sttStream = _sttRecognizer?.createStream();
      if (await _audioRecorder.hasPermission()) {
        setState(() => _isRecording = true);

        // FIX: Using 'wav' encoder to prevent build errors.
        // It works on all devices safely.
        final stream = await _audioRecorder.startStream(const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          numChannels: 1,
        ));

        stream.listen((data) {
          final samples = convertBytesToFloat32(Uint8List.fromList(data));
          _sttStream!.acceptWaveform(samples: samples, sampleRate: _sampleRate);
          while (_sttRecognizer!.isReady(_sttStream!))
            _sttRecognizer!.decode(_sttStream!);
          final text = _sttRecognizer!.getResult(_sttStream!).text;
          bool isEnd = _sttRecognizer!.isEndpoint(_sttStream!);
          _updateUI(text, isEnd);
          if (isEnd) _sttRecognizer!.reset(_sttStream!);
        });
      }
    }
  }

  void _updateUI(String text, bool isEndpoint) {
    if (text.isNotEmpty) {
      String full = '$_sentenceIndex: $text\n$_lastRecognizedText';
      _controller.value = TextEditingValue(
          text: full, selection: TextSelection.collapsed(offset: full.length));
      if (isEndpoint) {
        _lastRecognizedText = full;
        _sentenceIndex++;
        _toggleRecording().then((_) => _speak(text));
      }
    }
  }

  Future<void> _speak(String text) async {
    if (_ttsEngine == null) return;
    final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
    if (audio.samples.isEmpty) return;
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/tts.wav');
    await file.writeAsBytes(_createWav(audio.samples, audio.sampleRate),
        flush: true);
    await _audioPlayer.play(DeviceFileSource(file.path));
  }

  Uint8List _createWav(Float32List samples, int rate) {
    int channels = 1;
    int size = 36 + samples.length * 2;
    var buffer = BytesBuilder();
    buffer.add('RIFF'.codeUnits);
    buffer.add(_int32(size));
    buffer.add('WAVEfmt '.codeUnits);
    buffer.add(_int32(16));
    buffer.add(_int16(1));
    buffer.add(_int16(channels));
    buffer.add(_int32(rate));
    buffer.add(_int32(rate * 2));
    buffer.add(_int16(2));
    buffer.add(_int16(16));
    buffer.add('data'.codeUnits);
    buffer.add(_int32(samples.length * 2));
    for (var s in samples) {
      double v = s * 1.5;
      if (v > 1.0) v = 1.0;
      if (v < -1.0) v = -1.0;
      buffer.add(_int16((v * 32767).toInt()));
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
      appBar: AppBar(title: const Text('Jarvis AI')),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Text(_statusText,
                style: const TextStyle(
                    fontWeight: FontWeight.bold, color: Colors.blue)),
            const SizedBox(height: 20),
            Expanded(
                child: TextField(
                    controller: _controller,
                    maxLines: null,
                    readOnly: true,
                    decoration:
                        const InputDecoration(border: OutlineInputBorder()))),
            const SizedBox(height: 30),
            GestureDetector(
              onTap: _toggleRecording,
              child: CircleAvatar(
                radius: 40,
                backgroundColor: _isRecording ? Colors.red : Colors.blue,
                child: Icon(_isRecording ? Icons.stop : Icons.mic,
                    color: Colors.white, size: 40),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
