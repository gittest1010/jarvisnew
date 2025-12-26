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

// --- UTILITY: Convert Raw Audio Bytes to Float32 for Sherpa ---
Float32List convertBytesToFloat32(Uint8List bytes) {
  final int length = bytes.length ~/ 2;
  final Float32List float32List = Float32List(length);
  final ByteData byteData = ByteData.sublistView(bytes);

  for (int i = 0; i < length; i++) {
    // 16-bit PCM (Little Endian)
    int sample = byteData.getInt16(i * 2, Endian.little);
    // Normalize to -1.0 to 1.0
    float32List[i] = sample / 32768.0;
  }
  return float32List;
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: VoiceAssistantScreen()
  ));
}

class VoiceAssistantScreen extends StatefulWidget {
  const VoiceAssistantScreen({super.key});

  @override
  State<VoiceAssistantScreen> createState() => _VoiceAssistantScreenState();
}

class _VoiceAssistantScreenState extends State<VoiceAssistantScreen> {
  final TextEditingController _controller = TextEditingController();
  String _statusText = "Initializing...";
  
  // Audio Objects
  late final AudioRecorder _audioRecorder;
  late final AudioPlayer _audioPlayer;
  
  // Sherpa Objects
  sherpa_onnx.OnlineRecognizer? _sttRecognizer;
  sherpa_onnx.OnlineStream? _sttStream;
  sherpa_onnx.OfflineTts? _ttsEngine;
  
  // State
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

  // --- STEP 1: PERMISSIONS & SETUP ---
  Future<void> _startSetupProcess() async {
    // 1. Ask Microphone Permission
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
      setState(() => _statusText = "Microphone permission denied ❌");
      return;
    }

    // 2. Initialize Engines
    await _initSherpaEngines();
  }

  // --- STEP 2: COPY ASSETS TO PHONE STORAGE ---
  // Ye function single files copy karta hai
  Future<String> _copyAssetToFile(String assetPath) async {
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final fileName = assetPath.split('/').last;
      final file = File('${docsDir.path}/$fileName');

      // Agar file pehle se hai to overwrite nahi karenge (Speed badhegi)
      if (!await file.exists()) {
        setState(() => _statusText = "Copying $fileName...");
        final data = await rootBundle.load(assetPath);
        final bytes = data.buffer.asUint8List();
        await file.writeAsBytes(bytes, flush: true);
      }
      return file.path;
    } catch (e) {
      print("Error copying file $assetPath: $e");
      throw Exception("Could not copy $assetPath");
    }
  }

  // Ye function poore folder (espeak-ng-data) ko copy karta hai
  Future<String> _copyAssetFolder(String assetFolderPath) async {
    final docsDir = await getApplicationDocumentsDirectory();
    final localFolder = Directory('${docsDir.path}/${assetFolderPath.split('/').last}');
    
    // Hamesha check karein taaki bar bar copy na ho
    if (await localFolder.exists()) {
       // Optional: Agar aapko lagta hai files corrupt hain to ise delete karke wapas copy kar sakte hain
       // await localFolder.delete(recursive: true);
       return localFolder.path;
    }
    
    await localFolder.create(recursive: true);

    try {
      // AssetManifest load karein taaki folder ke andar ki files mil sakein
      final manifestContent = await rootBundle.loadString('AssetManifest.json');
      final Map<String, dynamic> manifestMap = json.decode(manifestContent);

      // Sirf wo files filter karein jo is folder mein hain
      final filePaths = manifestMap.keys
          .where((String key) => key.contains('espeak-ng-data'))
          .toList();

      if (filePaths.isEmpty) {
        throw Exception("espeak-ng-data folder empty hai! Pubspec check karein.");
      }

      for (final filePath in filePaths) {
        // Path fix karein (assets/espeak-ng-data/lang -> local/espeak-ng-data/lang)
        // Hame sirf folder ke baad ka hissa chahiye
        final relativePathIndex = filePath.indexOf('espeak-ng-data/');
        if (relativePathIndex == -1) continue;
        
        final relativePath = filePath.substring(relativePathIndex); 
        final localFilePath = '${docsDir.path}/$relativePath';
        final localFile = File(localFilePath);

        if (!await localFile.exists()) {
          setState(() => _statusText = "Unpacking data...");
          // Parent folder banayein agar nahi hai
          await localFile.parent.create(recursive: true);
          
          final data = await rootBundle.load(filePath);
          await localFile.writeAsBytes(data.buffer.asUint8List(), flush: true);
        }
      }
    } catch (e) {
      print("Folder Copy Error: $e");
      // Agar fail bhi ho, to path return kar dete hain, shayad files pehle se hon
    }
    
    return localFolder.path;
  }

  // --- STEP 3: INITIALIZE SHERPA ---
  Future<void> _initSherpaEngines() async {
    try {
      setState(() => _statusText = "Loading AI Models...");

      // A. Copy STT Files
      final encoderPath = await _copyAssetToFile('assets/tiny-encoder.int8.onnx');
      final decoderPath = await _copyAssetToFile('assets/tiny-decoder.int8.onnx');
      final tokensPath = await _copyAssetToFile('assets/tokens.txt');

      // B. Copy TTS Files
      final ttsModelPath = await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx');
      final ttsJsonPath = await _copyAssetToFile('assets/hi_IN-pratham-medium.onnx.json');
      
      // C. Copy Data Folder (The Big One)
      // Dhyan dein: 'assets/espeak-ng-data' pass kar rahe hain
      final espeakDataPath = await _copyAssetFolder('assets/espeak-ng-data');

      // D. Configure STT (Sunne wala)
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

      // E. Configure TTS (Bolne wala)
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: ttsModelPath,
            tokens: ttsJsonPath,
            dataDir: espeakDataPath, // Local path pass kiya
          ),
          provider: 'sherpa-onnx',
        ),
      );
      _ttsEngine = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() {
        _isInitialized = true;
        _statusText = "Jarvis Ready. Tap Mic 🎙️";
      });

    } catch (e) {
      setState(() => _statusText = "Initialization Failed: $e");
      print(e);
    }
  }

  // --- RECORDING LOGIC ---
  Future<void> _toggleRecording() async {
    if (!_isInitialized) return;

    if (_isRecording) {
      // Stop
      await _audioRecorder.stop();
      setState(() => _isRecording = false);
    } else {
      // Start
      _sttStream?.free();
      _sttStream = _sttRecognizer?.createStream();

      if (await _audioRecorder.hasPermission()) {
        setState(() => _isRecording = true);
        
        final stream = await _audioRecorder.startStream(const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
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

          if (isEndpoint) {
            _sttRecognizer!.reset(_sttStream!);
          }
        });
      }
    }
  }

  void _updateUI(String text, bool isEndpoint) {
    if (text.isNotEmpty) {
      String fullText = '$_sentenceIndex: $text\n$_lastRecognizedText';
      // UI Update
      _controller.value = TextEditingValue(
        text: fullText,
        selection: TextSelection.collapsed(offset: fullText.length),
      );

      if (isEndpoint) {
        _lastRecognizedText = fullText;
        _sentenceIndex++;
        // Stop Mic temporarily to avoid echo
        _toggleRecording().then((_) {
          _speak(text);
        });
      }
    }
  }

  // --- SPEAKING LOGIC (With AudioPlayer) ---
  Future<void> _speak(String text) async {
    if (_ttsEngine == null || text.isEmpty) return;
    print("Speaking: $text");

    // 1. Generate Audio Samples
    final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
    if (audio.samples.isEmpty) return;

    // 2. Convert to WAV File
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/tts_output.wav');
    
    // Create valid WAV header
    final wavBytes = _createWavHeader(audio.samples, audio.sampleRate);
    await file.writeAsBytes(wavBytes, flush: true);

    // 3. Play
    await _audioPlayer.play(DeviceFileSource(file.path));
  }

  // WAV Header Helper
  Uint8List _createWavHeader(Float32List samples, int sampleRate) {
    int numChannels = 1;
    int bitsPerSample = 16;
    int byteRate = sampleRate * numChannels * bitsPerSample ~/ 8;
    int blockAlign = numChannels * bitsPerSample ~/ 8;
    int dataSize = samples.length * numChannels * bitsPerSample ~/ 8;
    int chunkSize = 36 + dataSize;

    var buffer = BytesBuilder();
    // RIFF
    buffer.add('RIFF'.codeUnits);
    buffer.add(_int32(chunkSize));
    buffer.add('WAVE'.codeUnits);
    // fmt
    buffer.add('fmt '.codeUnits);
    buffer.add(_int32(16));
    buffer.add(_int16(1)); // PCM
    buffer.add(_int16(numChannels));
    buffer.add(_int32(sampleRate));
    buffer.add(_int32(byteRate));
    buffer.add(_int16(blockAlign));
    buffer.add(_int16(bitsPerSample));
    // data
    buffer.add('data'.codeUnits);
    buffer.add(_int32(dataSize));

    // Samples
    for (var sample in samples) {
      // Amplify volume slightly (x1.5) and clip
      double s = sample * 1.5;
      if (s > 1.0) s = 1.0;
      if (s < -1.0) s = -1.0;
      buffer.add(_int16((s * 32767).toInt()));
    }

    return buffer.toBytes();
  }

  List<int> _int32(int v) => Uint8List(4)..buffer.asByteData().setInt32(0, v, Endian.little);
  List<int> _int16(int v) => Uint8List(2)..buffer.asByteData().setInt16(0, v, Endian.little);

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
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: const Text('Jarvis AI'),
        backgroundColor: Colors.blueAccent,
        elevation: 0,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // Status Card
            Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(15),
                boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 10)]
              ),
              child: Row(
                children: [
                  Icon(Icons.info_outline, color: Colors.blueAccent),
                  SizedBox(width: 10),
                  Expanded(child: Text(_statusText, style: TextStyle(fontWeight: FontWeight.bold))),
                ],
              ),
            ),
            const SizedBox(height: 20),
            
            // Text Area
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(15),
                  border: Border.all(color: Colors.blueAccent.withOpacity(0.3))
                ),
                child: TextField(
                  controller: _controller,
                  maxLines: null,
                  readOnly: true,
                  style: TextStyle(fontSize: 18),
                  decoration: const InputDecoration(
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.all(20),
                    hintText: 'Recognized text will appear here...',
                  ),
                ),
              ),
            ),
            const SizedBox(height: 30),
            
            // Mic Button
            GestureDetector(
              onTap: _toggleRecording,
              child: AnimatedContainer(
                duration: Duration(milliseconds: 300),
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: _isRecording ? Colors.redAccent : Colors.blueAccent,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: (_isRecording ? Colors.redAccent : Colors.blueAccent).withOpacity(0.4),
                      blurRadius: 20,
                      spreadRadius: 5
                    )
                  ],
                ),
                child: Icon(
                  _isRecording ? Icons.stop : Icons.mic,
                  color: Colors.white,
                  size: 40,
                ),
              ),
            ),
            SizedBox(height: 10),
            Text(_isRecording ? "Listening..." : "Tap to Speak", style: TextStyle(color: Colors.grey[600])),
          ],
        ),
      ),
    );
  }
}