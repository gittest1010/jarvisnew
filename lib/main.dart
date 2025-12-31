import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';
import 'package:archive/archive_io.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:permission_handler/permission_handler.dart';

// --- NEW IMPORTS (Fixes Errors) ---
import 'package:audioplayers/audioplayers.dart';
import 'package:record/record.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Jarvis AI',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF121212),
        colorScheme: const ColorScheme.dark(primary: Colors.cyanAccent),
        useMaterial3: true,
      ),
      home: const InitScreen(),
    );
  }
}

/* ================= 1. INIT SCREEN (OLD LOGIC - NO CHANGE) ================= */

class InitScreen extends StatefulWidget {
  const InitScreen({super.key});
  @override
  State<InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<InitScreen> {
  String status = "Initializing Neural Core...";
  String logs = "";
  bool isError = false;

  @override
  void initState() {
    super.initState();
    _startSetup();
  }

  void _log(String msg, {bool error = false}) {
    debugPrint(msg);
    if (mounted) {
      setState(() {
        status = msg;
        logs += "\n$msg";
        if (error) isError = true;
      });
    }
  }

  Future<void> _startSetup() async {
    try {
      _log("Step 1: Permissions...");
      await Permission.microphone.request();

      _log("Step 2: Native Bindings...");
      sherpa_onnx.initBindings();

      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;
      _log("Root: $basePath");

      // --- EXTRACTION ---
      await _extractIfNeeded("assets/stt-hi.tar.bz2", "stt_root", basePath);
      await _extractIfNeeded("assets/tts-hi.tar.bz2", "tts_root", basePath);

      // --- SMART FINDING ---
      _log("Step 3: Searching for Model Files...");

      final sttDir = Directory("$basePath/stt_root");
      final ttsDir = Directory("$basePath/tts_root");

      // 1. Find STT Files
      final encoder = await _recursiveFind(sttDir, "tiny-encoder.int8.onnx");
      final decoder = await _recursiveFind(sttDir, "tiny-decoder.int8.onnx");

      // Check for tokens.txt OR tokens.text
      var sttTokens = await _recursiveFind(sttDir, "tokens.txt");
      if (sttTokens == null) {
        _log("tokens.txt not found, checking tokens.text...");
        sttTokens = await _recursiveFind(sttDir, "tokens.text");
      }

      // 2. Find TTS Files
      final ttsModel = await _recursiveFind(ttsDir, "model.onnx");

      var ttsTokens = await _recursiveFind(ttsDir, "tokens.txt");
      if (ttsTokens == null) {
        ttsTokens = await _recursiveFind(ttsDir, "tokens.text");
      }

      final espeakData =
          await _recursiveFind(ttsDir, "espeak-ng-data", isFolder: true);

      // --- VALIDATION ---
      if (encoder == null ||
          decoder == null ||
          sttTokens == null ||
          ttsModel == null ||
          ttsTokens == null ||
          espeakData == null) {
        throw "Critical Asset Missing. Files not found after extraction.";
      }

      _log("All files located successfully!");

      await Future.delayed(const Duration(seconds: 1));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
              builder: (_) => JarvisHome(paths: {
                    "encoder": encoder,
                    "decoder": decoder,
                    "sttTokens": sttTokens!,
                    "ttsModel": ttsModel,
                    "ttsTokens": ttsTokens!,
                    "espeakData": espeakData,
                  })),
        );
      }
    } catch (e, stack) {
      _log("CRITICAL ERROR: $e", error: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  // Recursive Finder (Original Logic)
  Future<String?> _recursiveFind(Directory dir, String filename,
      {bool isFolder = false}) async {
    try {
      if (!await dir.exists()) return null;

      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity.path.endsWith("/$filename") ||
            entity.path.endsWith("\\$filename")) {
          if (isFolder && entity is Directory) {
            if (filename == "espeak-ng-data") {
              if (File("${entity.path}/phontab").existsSync())
                return entity.path;
            } else {
              return entity.path;
            }
          } else if (!isFolder && entity is File) {
            if (entity.lengthSync() > 100) return entity.path;
          }
        }
      }
    } catch (e) {
      print("Search error: $e");
    }
    return null;
  }

  Future<void> _extractIfNeeded(
      String asset, String folderName, String basePath) async {
    final target = Directory("$basePath/$folderName");
    if (await target.exists() && target.listSync().isNotEmpty) {
      return;
    }
    _log("Extracting $asset...");
    try {
      final data = await rootBundle.load(asset);
      final bytes = data.buffer.asUint8List();
      await compute(_backgroundUnzip, _UnzipArgs(bytes, basePath, folderName));
    } catch (e) {
      throw "Asset not found: $asset";
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (!isError)
              const CircularProgressIndicator(color: Colors.cyanAccent),
            const SizedBox(height: 20),
            Text(status,
                textAlign: TextAlign.center,
                style: TextStyle(color: isError ? Colors.red : Colors.white)),
            if (isError)
              Container(
                height: 200,
                padding: const EdgeInsets.all(10),
                child: SingleChildScrollView(
                    child: Text(logs, style: const TextStyle(fontSize: 10))),
              )
          ],
        ),
      ),
    );
  }
}

class _UnzipArgs {
  final Uint8List bytes;
  final String basePath;
  final String targetFolder;
  _UnzipArgs(this.bytes, this.basePath, this.targetFolder);
}

Future<void> _backgroundUnzip(_UnzipArgs args) async {
  final archive =
      TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(args.bytes));
  for (final file in archive) {
    final filename = "${args.basePath}/${args.targetFolder}/${file.name}";
    if (file.isFile) {
      final f = File(filename);
      if (!f.parent.existsSync()) f.parent.createSync(recursive: true);
      f.writeAsBytesSync(file.content as List<int>);
    } else {
      Directory(filename).createSync(recursive: true);
    }
  }
}

/* ================= 2. JARVIS HOME (NEW INTELLIGENT MODE) ================= */

class JarvisHome extends StatefulWidget {
  final Map<String, String> paths;
  const JarvisHome({super.key, required this.paths});

  @override
  State<JarvisHome> createState() => _JarvisHomeState();
}

class _JarvisHomeState extends State<JarvisHome> {
  // Engines
  sherpa_onnx.OnlineRecognizer? recognizer;
  sherpa_onnx.OfflineTts? tts;

  // Audio Components
  final AudioRecorder _audioRecorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription? _recordingSubscription;

  // State
  String userText = "Tap mic to speak...";
  String aiText = "System Initialized.";
  bool isListening = false;
  bool isProcessing = false;
  bool isSpeaking = false;

  @override
  void initState() {
    super.initState();
    _initAI();
  }

  void _initAI() {
    try {
      // 1. Init STT (Hearing)
      recognizer = sherpa_onnx.OnlineRecognizer(
        sherpa_onnx.OnlineRecognizerConfig(
          model: sherpa_onnx.OnlineModelConfig(
            transducer: sherpa_onnx.OnlineTransducerModelConfig(
              encoder: widget.paths["encoder"]!,
              decoder: widget.paths["decoder"]!,
              joiner: widget.paths["sttTokens"]!,
            ),
            tokens: widget.paths["sttTokens"]!,
            numThreads: 1,
          ),
          ruleFsts: "",
        ),
      );

      // 2. Init TTS (Speaking)
      tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
          model: sherpa_onnx.OfflineTtsModelConfig(
            vits: sherpa_onnx.OfflineTtsVitsModelConfig(
              model: widget.paths["ttsModel"]!,
              tokens: widget.paths["ttsTokens"]!,
              dataDir: widget.paths["espeakData"]!,
            ),
            provider: 'sherpa-onnx',
            numThreads: 1,
          ),
        ),
      );

      _greetUser();
    } catch (e) {
      setState(() => aiText = "Error loading AI: $e");
    }
  }

  Future<void> _greetUser() async {
    await Future.delayed(const Duration(seconds: 1));
    _speak("I am online.");
  }

  // --- LISTENING LOGIC (STT) ---

  Future<void> _toggleListening() async {
    // If speaking, stop speaking
    if (isSpeaking) {
      await _audioPlayer.stop();
      setState(() => isSpeaking = false);
    }

    if (isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    if (await _audioRecorder.hasPermission()) {
      try {
        setState(() {
          isListening = true;
          userText = "Listening...";
          aiText = "...";
        });

        recognizer?.reset();

        // Start Stream (16kHz, Mono, PCM)
        final stream = await _audioRecorder.startStream(
          const RecordConfig(
            encoder: AudioEncoder.pcm16bits,
            sampleRate: 16000,
            numChannels: 1,
          ),
        );

        _recordingSubscription = stream.listen((data) {
          _processAudioChunk(data);
        });
      } catch (e) {
        setState(() => aiText = "Mic Error: $e");
      }
    }
  }

  void _processAudioChunk(Uint8List data) {
    if (recognizer == null) return;

    // Convert Int16 bytes to Float32 array (-1.0 to 1.0)
    final int16List = Int16List.view(data.buffer);
    final float32List = Float32List(int16List.length);
    for (int i = 0; i < int16List.length; i++) {
      float32List[i] = int16List[i] / 32768.0;
    }

    // Feed to Engine
    recognizer!.acceptWaveform(float32List, 16000);

    // Get Live Result
    final result = recognizer!.getResult();
    if (result.text.isNotEmpty) {
      setState(() => userText = result.text);
    }

    // Auto-Stop on Silence (Endpoint Detection)
    if (recognizer!.isEndpoint()) {
      _stopListening();
    }
  }

  Future<void> _stopListening() async {
    await _audioRecorder.stop();
    await _recordingSubscription?.cancel();
    setState(() => isListening = false);

    if (recognizer != null) {
      final finalResult = recognizer!.getResult();
      final text = finalResult.text;

      if (text.isNotEmpty) {
        setState(() => userText = text);
        _processQuery(text); // Send to AI Brain
      } else {
        setState(() => userText = "Listening stopped.");
      }
    }
  }

  // --- AI BRAIN (SIMPLE LOGIC) ---

  void _processQuery(String query) {
    setState(() => isProcessing = true);

    // Simple Rule-Based AI (Offline)
    String response = "";
    final q = query.toLowerCase();

    if (q.contains("नमस्ते") || q.contains("hello") || q.contains("hi")) {
      response = "नमस्ते! मैं जार्विस हूँ।";
    } else if (q.contains("kaise ho") || q.contains("kaise hain")) {
      response = "मैं एक एआई हूँ, इसलिए मुझे थकान नहीं होती।";
    } else if (q.contains("time") || q.contains("samay")) {
      final now = DateTime.now();
      response = "अभी का समय है ${now.hour} बजकर ${now.minute} मिनट।";
    } else {
      // Echo response
      response = "आपने कहा: $query";
    }

    setState(() {
      aiText = response;
      isProcessing = false;
    });

    _speak(response); // Auto-Speak
  }

  // --- SPEAKING LOGIC (TTS) ---

  Future<void> _speak(String text) async {
    if (tts == null) return;

    setState(() => isSpeaking = true);

    try {
      final audio = tts!.generate(text: text, sid: 0, speed: 1.0);

      // Convert to WAV
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/tts_output.wav');
      await _writeWavFile(file, audio.samples, audio.sampleRate);

      // Play
      await _audioPlayer.play(DeviceFileSource(file.path));

      _audioPlayer.onPlayerComplete.listen((_) {
        if (mounted) setState(() => isSpeaking = false);
      });
    } catch (e) {
      print("TTS Error: $e");
      setState(() => isSpeaking = false);
    }
  }

  // Helper for WAV header
  Future<void> _writeWavFile(
      File file, Float32List samples, int sampleRate) async {
    final int numSamples = samples.length;
    final int byteRate = sampleRate * 2;
    var header = ByteData(44);
    var offset = 0;

    void writeString(String s) {
      for (int i = 0; i < s.length; i++)
        header.setUint8(offset + i, s.codeUnitAt(i));
      offset += s.length;
    }

    writeString('RIFF');
    header.setUint32(offset, 36 + numSamples * 2, Endian.little);
    offset += 4;
    writeString('WAVE');
    writeString('fmt ');
    header.setUint32(offset, 16, Endian.little);
    offset += 4;
    header.setUint16(offset, 1, Endian.little);
    offset += 2;
    header.setUint16(offset, 1, Endian.little);
    offset += 2;
    header.setUint32(offset, sampleRate, Endian.little);
    offset += 4;
    header.setUint32(offset, byteRate, Endian.little);
    offset += 4;
    header.setUint16(offset, 2, Endian.little);
    offset += 2;
    header.setUint16(offset, 16, Endian.little);
    offset += 2;
    writeString('data');
    header.setUint32(offset, numSamples * 2, Endian.little);
    offset += 4;

    final pcmBytes = Int16List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      var s = samples[i];
      if (s > 1.0)
        s = 1.0;
      else if (s < -1.0) s = -1.0;
      pcmBytes[i] = (s * 32767).toInt();
    }

    final builder = BytesBuilder();
    builder.add(header.buffer.asUint8List());
    builder.add(pcmBytes.buffer.asUint8List());
    await file.writeAsBytes(builder.toBytes());
  }

  @override
  void dispose() {
    _recordingSubscription?.cancel();
    _audioRecorder.dispose();
    _audioPlayer.dispose();
    recognizer?.free();
    tts?.free();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("JARVIS AI"),
        backgroundColor: Colors.transparent,
        elevation: 0,
        centerTitle: true,
      ),
      body: Column(
        children: [
          // 1. AI Response Area
          Expanded(
            flex: 4,
            child: Container(
              margin: const EdgeInsets.all(20),
              padding: const EdgeInsets.all(20),
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[900],
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: Colors.cyanAccent.withOpacity(0.3)),
              ),
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text("JARVIS:",
                        style: TextStyle(
                            color: Colors.cyanAccent,
                            fontSize: 12,
                            fontWeight: FontWeight.bold)),
                    const SizedBox(height: 10),
                    Text(
                      aiText,
                      style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w300,
                          color: Colors.white),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // 2. User Speech Area
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 30),
              child: Text(
                userText,
                textAlign: TextAlign.center,
                style: TextStyle(
                    fontSize: 18,
                    color: isListening ? Colors.greenAccent : Colors.grey,
                    fontStyle: FontStyle.italic),
              ),
            ),
          ),

          // 3. Controls (Mic Button)
          Expanded(
            flex: 3,
            child: Center(
              child: GestureDetector(
                onTap: _toggleListening,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  height: isListening ? 100 : 80,
                  width: isListening ? 100 : 80,
                  decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      // Red if listening, Cyan if idle
                      color: isListening
                          ? Colors.redAccent
                          : Colors.cyanAccent.withOpacity(0.1),
                      border: Border.all(
                        color: isListening ? Colors.red : Colors.cyanAccent,
                        width: isListening ? 5 : 2,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: (isListening ? Colors.red : Colors.cyanAccent)
                              .withOpacity(0.4),
                          blurRadius: isListening ? 30 : 10,
                          spreadRadius: isListening ? 5 : 0,
                        )
                      ]),
                  child: Icon(
                    isListening ? Icons.mic : Icons.mic_none,
                    size: 40,
                    color: isListening ? Colors.white : Colors.cyanAccent,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
