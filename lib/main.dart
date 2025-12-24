import 'dart:async';
import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

// --- UTILITY FUNCTION (To avoid needing utils.dart) ---
Float32List convertBytesToFloat32(Uint8List bytes) {
  final int length = bytes.length ~/ 2;
  final Float32List float32List = Float32List(length);
  final ByteData byteData = ByteData.sublistView(bytes);

  for (int i = 0; i < length; i++) {
    // Assuming 16-bit PCM little-endian
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
    
    // Listen to recording state changes
    _recordSub = _audioRecorder.onStateChanged().listen((recordState) {
      setState(() => _recordState = recordState);
    });

    // Initialize Sherpa (STT + TTS)
    _initSherpa();
  }

  // --- INITIALIZATION LOGIC ---
  Future<void> _initSherpa() async {
    try {
      // 1. Initialize Speech-to-Text (STT) - Whisper Tiny
      final sttModelConfig = sherpa_onnx.OnlineModelConfig(
        transducer: sherpa_onnx.OnlineTransducerModelConfig(
          encoder: 'assets/tiny-encoder.int8.onnx',
          decoder: 'assets/tiny-decoder.int8.onnx',
          joiner: 'assets/tokens.txt',
        ),
      );
      
      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sttModelConfig,
        ruleFsts: '',
      );
      
      _sttRecognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      // 2. Initialize Text-to-Speech (TTS) - Piper Pratham
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: 'assets/hi_IN-pratham-medium.onnx',
            tokens: 'assets/hi_IN-pratham-medium.onnx.json',
            dataDir: 'assets/espeak-ng-data',
          ),
          provider: 'sherpa-onnx',
        ),
      );
      
      _ttsEngine = sherpa_onnx.OfflineTts(config: ttsConfig);

      setState(() {
        _isInitialized = true;
        _statusText = "Ready. Press Mic to Speak.";
      });
      
    } catch (e) {
      setState(() {
        _statusText = "Error loading models: $e";
      });
      print("Error Initializing Sherpa: $e");
    }
  }

  // --- RECORDING & STT LOGIC ---
  Future<void> _startRecording() async {
    if (!_isInitialized) return;

    // Create a new stream for the recognizer
    _sttStream?.free();
    _sttStream = _sttRecognizer?.createStream();

    try {
      if (await _audioRecorder.hasPermission()) {
        const config = RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
        );

        // Start recording to stream
        final stream = await _audioRecorder.startStream(config);

        stream.listen(
          (data) {
            // Process audio data
            final samplesFloat32 = convertBytesToFloat32(Uint8List.fromList(data));
            
            _sttStream!.acceptWaveform(samples: samplesFloat32, sampleRate: _sampleRate);
            
            while (_sttRecognizer!.isReady(_sttStream!)) {
              _sttRecognizer!.decode(_sttStream!);
            }
            
            final text = _sttRecognizer!.getResult(_sttStream!).text;
            
            // UI Update Logic
            bool isEndpoint = _sttRecognizer!.isEndpoint(_sttStream!);
            _updateUiWithText(text, isEndpoint);

            if (isEndpoint) {
              _sttRecognizer!.reset(_sttStream!);
            }
          },
          onDone: () {
            print('Recording stopped');
          },
        );
      }
    } catch (e) {
      print("Error starting record: $e");
    }
  }

  Future<void> _stopRecording() async {
    await _audioRecorder.stop();
    _sttStream?.free();
    _sttStream = null;
  }

  void _updateUiWithText(String currentText, bool isEndpoint) {
    if (currentText.isNotEmpty) {
      String display = _lastRecognizedText;
      if (display.isEmpty) {
        display = '$_sentenceIndex: $currentText';
      } else {
        display = '$_sentenceIndex: $currentText\n$_lastRecognizedText';
      }

      _controller.value = TextEditingValue(
        text: display,
        selection: TextSelection.collapsed(offset: display.length),
      );
      
      // If sentence finished (Endpoint), Speak it back (TTS)
      if (isEndpoint) {
        _lastRecognizedText = display;
        _sentenceIndex++;
        // Trigger TTS
        _speakText(currentText);
      }
    }
  }

  // --- TTS LOGIC ---
  Future<void> _speakText(String text) async {
    if (_ttsEngine == null || text.isEmpty) return;

    print("Generating Audio for: $text");
    
    // Generate Audio
    final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
    
    // NOTE: In a real app, you would use 'audioplayers' or 'soundpool' package 
    // to play 'audio.samples'. Since we are keeping code simple, we will 
    // save it to a file to prove it works.
    
    if (audio.samples.isNotEmpty) {
       print("TTS Generated ${audio.samples.length} samples. (Add AudioPlayer to hear it)");
       // Code to save wav file (Optional verification)
       // final dir = await getTemporaryDirectory();
       // final file = File('${dir.path}/tts_output.wav');
       // ... write wav header and samples ...
    }
  }

  @override
  void dispose() {
    _recordSub?.cancel();
    _audioRecorder.dispose();
    _sttStream?.free();
    _sttRecognizer?.free();
    _ttsEngine?.free();
    _controller.dispose();
    super.dispose();
  }

  // --- UI BUILD ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Sherpa ONNX Assistant')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(_statusText, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
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