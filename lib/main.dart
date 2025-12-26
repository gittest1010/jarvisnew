import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // For rootBundle
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart'; // For Playing Audio

// --- UTILITY FUNCTION ---
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
  runApp(const MaterialApp(home: VoiceAssistantScreen()));
}

class VoiceAssistantScreen extends StatefulWidget {
  const VoiceAssistantScreen({super.key});

  @override
  State<VoiceAssistantScreen> createState() => _VoiceAssistantScreenState();
}

class _VoiceAssistantScreenState extends State<VoiceAssistantScreen> {
  // UI Controllers
  late final TextEditingController _controller;
  String _statusText = "Initializing...";
  
  // Audio Recording
  late final AudioRecorder _audioRecorder;
  StreamSubscription<RecordState>? _recordSub;
  RecordState _recordState = RecordState.stop;

  // Audio Playing
  late final AudioPlayer _audioPlayer;

  // Sherpa ONNX Engines
  sherpa_onnx.OnlineRecognizer? _sttRecognizer;
  sherpa_onnx.OnlineStream? _sttStream;
  sherpa_onnx.OfflineTts? _ttsEngine;
  
  // State Variables
  bool _isInitialized = false;
  String _lastRecognizedText = '';
  int _sentenceIndex = 0;
  final int _sampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController();
    _audioRecorder = AudioRecorder();
    _audioPlayer = AudioPlayer(); // Initialize Player
    
    _recordSub = _audioRecorder.onStateChanged().listen((recordState) {
      setState(() => _recordState = recordState);
    });

    _requestPermissionsAndInit();
  }

  Future<void> _requestPermissionsAndInit() async {
    // Permissions maangna zaruri hai
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() => _statusText = "Microphone permission denied!");
      return;
    }
    await _initSherpa();
  }

  // --- ASSET COPYING LOGIC (FILES) ---
  Future<String> _copyAssetToFile(String assetPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final fileName = assetPath.split('/').last;
    final file = File('${docsDir.path}/$fileName');

    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      final bytes = data.buffer.asUint8List();
      await file.writeAsBytes(bytes);
    }
    return file.path;
  }

  // --- ASSET COPYING LOGIC (FOLDERS) - Fixes espeak-ng-data error ---
  Future<String> _copyAssetFolder(String assetFolderPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final localDir = Directory('${docsDir.path}/${assetFolderPath.split('/').last}');
    
    if (!await localDir.exists()) {
      await localDir.create(recursive: true);
    }

    // Load AssetManifest to find all files in the folder
    final manifestContent = await rootBundle.loadString('AssetManifest.json');
    final Map<String, dynamic> manifestMap = json.decode(manifestContent);

    // Filter files belonging to this folder
    final filePaths = manifestMap.keys
        .where((String key) => key.startsWith(assetFolderPath))
        .toList();

    for (final filePath in filePaths) {
      final relativePath = filePath.substring(assetFolderPath.length);
      final localFilePath = '${localDir.path}/$relativePath';
      final localFile = File(localFilePath);

      if (!await localFile.exists()) {
        final parentDir = localFile.parent;
        if (!await parentDir.exists()) {
          await parentDir.create(recursive: true);
        }
        final data = await rootBundle.load(filePath);
        await localFile.writeAsBytes(data.buffer.asUint8List());
      }
    }
    
    return localDir.path;
  }

  // --- INITIALIZATION ---
  Future<void> _initSherpa() async {
    try {
      setState(() => _statusText = "Copying Models (First Time)...");

      // 1. Copy STT Models
      final encoderPath = await _copyAssetToFile('assets/tiny-encoder.int8.onnx');
      final decoderPath = await _copyAssetToFile('assets/tiny-decoder.int8.onnx');
      final tokensPath = await _copyAssetToFile('assets/tokens.txt');
      
      // 2. Copy TTS Models & Data
      final ttsModelPath = await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx');
      final ttsTokensPath = await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx.json');
      
      // CRITICAL FIX: Copy espeak-ng-data folder recursively
      final espeakDataPath = await _copyAssetFolder('assets/espeak-ng-data');

      // 3. Initialize STT
      final sttModelConfig = sherpa_onnx.OnlineModelConfig(
        transducer: sherpa_onnx.OnlineTransducerModelConfig(
          encoder: encoderPath,
          decoder: decoderPath,
          joiner: tokensPath,
        ),
        tokens: tokensPath,
      );
      
      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sttModelConfig,
        ruleFsts: '',
      );
      
      _sttRecognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      // 4. Initialize TTS
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: ttsModelPath,
            tokens: ttsTokensPath,
            dataDir: espeakDataPath, // Using valid local path
          ),
          provider: 'sherpa-onnx',
        ),
      );
      
      _ttsEngine = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() {
        _isInitialized = true;
        _statusText = "Ready. Press Mic to Speak.";
      });
      
    } catch (e) {
      setState(() => _statusText = "Error: $e");
      print("Error Initializing Sherpa: $e");
    }
  }

  // --- RECORDING ---
  Future<void> _startRecording() async {
    if (!_isInitialized) return;

    _sttStream?.free();
    _sttStream = _sttRecognizer?.createStream();

    try {
      if (await _audioRecorder.hasPermission()) {
        const config = RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
        );
        final stream = await _audioRecorder.startStream(config);

        stream.listen(
          (data) {
            final samplesFloat32 = convertBytesToFloat32(Uint8List.fromList(data));
            _sttStream!.acceptWaveform(samples: samplesFloat32, sampleRate: _sampleRate);
            
            while (_sttRecognizer!.isReady(_sttStream!)) {
              _sttRecognizer!.decode(_sttStream!);
            }
            
            final text = _sttRecognizer!.getResult(_sttStream!).text;
            bool isEndpoint = _sttRecognizer!.isEndpoint(_sttStream!);
            
            _updateUiWithText(text, isEndpoint);

            if (isEndpoint) {
              _sttRecognizer!.reset(_sttStream!);
            }
          },
        );
      }
    } catch (e) {
      print("Record Error: $e");
    }
  }

  Future<void> _stopRecording() async {
    await _audioRecorder.stop();
  }

  void _updateUiWithText(String currentText, bool isEndpoint) {
    if (currentText.isNotEmpty) {
      String fullText = '$_sentenceIndex: $currentText';
      if (_lastRecognizedText.isNotEmpty) {
         fullText = '$fullText\n$_lastRecognizedText';
      }

      _controller.value = TextEditingValue(
        text: fullText,
        selection: TextSelection.collapsed(offset: fullText.length),
      );
      
      if (isEndpoint) {
        _lastRecognizedText = fullText;
        _sentenceIndex++;
        // Stop recording temporarily while speaking to avoid echo
        _stopRecording().then((_) => _speakText(currentText));
      }
    }
  }

  // --- TTS PLAYBACK LOGIC (The Magic Part) ---
  Future<void> _speakText(String text) async {
    if (_ttsEngine == null || text.isEmpty) return;

    print("Generating Audio...");
    // 1. Generate Raw Audio
    final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
    
    if (audio.samples.isEmpty) return;

    // 2. Convert to WAV File
    final dir = await getTemporaryDirectory();
    final filePath = '${dir.path}/tts_output.wav';
    final wavFile = File(filePath);

    // Create WAV Header and Write Data
    final wavBytes = _createWavFile(audio.samples, audio.sampleRate);
    await wavFile.writeAsBytes(wavBytes);

    // 3. Play Immediately
    print("Playing Audio...");
    await _audioPlayer.play(DeviceFileSource(filePath));
    
    // Optional: Resume listening after speaking
    // _startRecording(); 
  }

  // --- WAV HEADER GENERATOR ---
  Uint8List _createWavFile(Float32List samples, int sampleRate) {
    int numSamples = samples.length;
    int numChannels = 1;
    int bitsPerSample = 16;
    int byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
    int blockAlign = numChannels * bitsPerSample ~/ 8;
    int subChunk2Size = numSamples * numChannels * bitsPerSample ~/ 8;
    int chunkSize = 36 + subChunk2Size;

    var buffer = BytesBuilder();

    // RIFF chunk
    buffer.add('RIFF'.codeUnits);
    buffer.add(_int32ToBytes(chunkSize));
    buffer.add('WAVE'.codeUnits);

    // fmt chunk
    buffer.add('fmt '.codeUnits);
    buffer.add(_int32ToBytes(16)); // SubChunk1Size
    buffer.add(_int16ToBytes(1)); // AudioFormat (PCM)
    buffer.add(_int16ToBytes(numChannels));
    buffer.add(_int32ToBytes(sampleRate));
    buffer.add(_int32ToBytes(byteRate));
    buffer.add(_int16ToBytes(blockAlign));
    buffer.add(_int16ToBytes(bitsPerSample));

    // data chunk
    buffer.add('data'.codeUnits);
    buffer.add(_int32ToBytes(subChunk2Size));

    // Write samples (Convert Float32 -1.0..1.0 to Int16)
    for (var sample in samples) {
      // Clip to -1.0 to 1.0
      if (sample > 1.0) sample = 1.0;
      if (sample < -1.0) sample = -1.0;
      
      int val = (sample * 32767).toInt();
      buffer.add(_int16ToBytes(val));
    }

    return buffer.toBytes();
  }

  List<int> _int32ToBytes(int value) {
    return Uint8List(4)..buffer.asByteData().setInt32(0, value, Endian.little);
  }

  List<int> _int16ToBytes(int value) {
    return Uint8List(2)..buffer.asByteData().setInt16(0, value, Endian.little);
  }

  @override
  void dispose() {
    _recordSub?.cancel();
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
      appBar: AppBar(title: const Text('Jarvis AI (Speaking)')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(_statusText, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.blue)),
            const SizedBox(height: 20),
            Expanded(
              child: TextField(
                controller: _controller,
                maxLines: null,
                readOnly: true,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  hintText: 'Recognized text will appear here...',
                ),
              ),
            ),
            const SizedBox(height: 30),
            GestureDetector(
              onTap: () {
                if (!_isInitialized) return;
                (_recordState == RecordState.stop) ? _startRecording() : _stopRecording();
              },
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: _recordState == RecordState.stop ? Colors.blue : Colors.red,
                  shape: BoxShape.circle,
                  boxShadow: [
                     BoxShadow(color: Colors.black26, blurRadius: 10, spreadRadius: 2)
                  ],
                ),
                child: Icon(
                  _recordState == RecordState.stop ? Icons.mic : Icons.stop,
                  color: Colors.white,
                  size: 40,
                ),
              ),
            ),
            const SizedBox(height: 20),
            Text(
              _recordState == RecordState.stop ? "Tap to Speak" : "Listening...",
              style: TextStyle(
                color: _recordState == RecordState.stop ? Colors.black : Colors.red,
                fontSize: 18
              ),
            ),
          ],
        ),
      ),
    );
  }
}