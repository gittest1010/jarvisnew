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
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.cyanAccent,
        brightness: Brightness.dark, // Futuristic Look
      ),
      home: const InitScreen(),
    );
  }
}

/* ================= 1. INITIALIZATION & VALIDATION SCREEN ================= */

class InitScreen extends StatefulWidget {
  const InitScreen({super.key});
  @override
  State<InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<InitScreen> {
  String status = "System Initializing...";
  String errorLog = "";
  bool isReady = false;
  bool isError = false;

  // Countdown variables
  bool isCountingDown = false;
  int countdown = 5;

  @override
  void initState() {
    super.initState();
    _startSystemCheck();
  }

  void _updateStatus(String msg) {
    debugPrint(msg);
    if (mounted) setState(() => status = msg);
  }

  void _showError(String error) {
    debugPrint("ERROR: $error");
    if (mounted) {
      setState(() {
        status = "System Failure";
        errorLog = error;
        isError = true;
      });
    }
  }

  Future<void> _startSystemCheck() async {
    try {
      // 1. Permissions
      _updateStatus("Checking Security Permissions...");
      await [Permission.microphone].request();

      // 2. Init Native Libs
      _updateStatus("Loading Neural Engine...");
      sherpa_onnx.initBindings();

      // 3. Path Setup
      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;

      // 4. Extraction (Only if needed)
      await _manageAsset("assets/stt-hi.tar.bz2", "stt_root", basePath);
      await _manageAsset("assets/tts-hi.tar.bz2", "tts_root", basePath);

      // 5. DEEP VALIDATION (The most important part)
      _updateStatus("Verifying Integrity...");
      await _validateFiles(basePath);

      // 6. Success
      if (mounted) {
        setState(() {
          status = "Systems Online";
          isReady = true;
        });
      }
    } catch (e, stack) {
      _showError("Critical Error:\n$e");
      debugPrintStack(stackTrace: stack);
    }
  }

  // Smart Extraction: Checks if folder is empty
  Future<void> _manageAsset(
      String assetPath, String folderName, String basePath) async {
    final targetDir = Directory("$basePath/$folderName");

    // Check if folder exists and has content
    if (await targetDir.exists()) {
      if (await targetDir.list().isEmpty) {
        _updateStatus("Repairing $folderName...");
      } else {
        _updateStatus("$folderName Found. Skipping extraction.");
        return; // Already exists
      }
    }

    _updateStatus("Extracting $folderName...");
    try {
      final data = await rootBundle.load(assetPath);
      final bytes = data.buffer.asUint8List();
      await compute(
          _backgroundExtract, _ExtractArgs(bytes, basePath, folderName));
    } catch (e) {
      throw Exception("Failed to extract $assetPath. Is the file in assets?");
    }
  }

  // Validation Logic: Checks specific files
  Future<void> _validateFiles(String basePath) async {
    final requiredFiles = {
      "STT Encoder": "$basePath/stt_root/tiny-encoder.int8.onnx",
      "STT Decoder": "$basePath/stt_root/tiny-decoder.int8.onnx",
      "STT Tokens": "$basePath/stt_root/tokens.txt",
      "TTS Model": "$basePath/tts_root/model.onnx",
      "TTS Tokens": "$basePath/tts_root/tokens.txt",
      "Espeak Phontab": "$basePath/tts_root/espeak-ng-data/phontab", // CRITICAL
    };

    for (var entry in requiredFiles.entries) {
      final file = File(entry.value);
      if (!await file.exists()) {
        throw Exception("MISSING FILE: ${entry.key}\nPath: ${entry.value}");
      }
      if (await file.length() < 100) {
        throw Exception(
            "CORRUPT FILE: ${entry.key}\nSize too small (<100B). Extraction failed.");
      }
    }
  }

  void _startCountdown() {
    setState(() {
      isCountingDown = true;
    });

    Timer.periodic(const Duration(seconds: 1), (timer) {
      if (countdown == 1) {
        timer.cancel();
        // Launch Main App
        Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (_) => const JarvisHome()));
      } else {
        if (mounted) {
          setState(() => countdown--);
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(30.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo or Icon
              const Icon(Icons.mic_external_on,
                  size: 80, color: Colors.cyanAccent),
              const SizedBox(height: 30),

              if (isCountingDown) ...[
                Text(
                  "$countdown",
                  style: const TextStyle(
                      fontSize: 80,
                      fontWeight: FontWeight.bold,
                      color: Colors.cyanAccent),
                ),
                const Text("Launching Core...",
                    style: TextStyle(color: Colors.white54)),
              ] else ...[
                Text(
                  status,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      fontSize: 18,
                      color: isError ? Colors.redAccent : Colors.white,
                      fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 20),
                if (!isReady && !isError)
                  const CircularProgressIndicator(color: Colors.cyanAccent),
                if (isError)
                  Container(
                    height: 200,
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.1),
                        border: Border.all(color: Colors.redAccent),
                        borderRadius: BorderRadius.circular(10)),
                    child: SingleChildScrollView(
                      child: Text(errorLog,
                          style: const TextStyle(
                              color: Colors.red, fontFamily: 'monospace')),
                    ),
                  ),
                if (isReady)
                  ElevatedButton(
                    onPressed: _startCountdown,
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.cyanAccent,
                        foregroundColor: Colors.black,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 40, vertical: 15),
                        textStyle: const TextStyle(
                            fontSize: 20, fontWeight: FontWeight.bold)),
                    child: const Text("START JARVIS"),
                  ),
              ]
            ],
          ),
        ),
      ),
    );
  }
}

// Background Isolate Logic
class _ExtractArgs {
  final Uint8List bytes;
  final String basePath;
  final String targetFolder;
  _ExtractArgs(this.bytes, this.basePath, this.targetFolder);
}

Future<void> _backgroundExtract(_ExtractArgs args) async {
  final archive =
      TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(args.bytes));
  for (final file in archive) {
    final cleanName = file.name.replaceAll("../", "");
    final filename = "${args.basePath}/${args.targetFolder}/$cleanName";
    if (file.isFile) {
      final f = File(filename);
      if (!f.parent.existsSync()) f.parent.createSync(recursive: true);
      f.writeAsBytesSync(file.content as List<int>);
    } else {
      Directory(filename).createSync(recursive: true);
    }
  }
}

/* ================= 2. MAIN APP LOGIC (JARVIS HOME) ================= */

class JarvisHome extends StatefulWidget {
  const JarvisHome({super.key});
  @override
  State<JarvisHome> createState() => _JarvisHomeState();
}

class _JarvisHomeState extends State<JarvisHome> {
  sherpa_onnx.OnlineRecognizer? recognizer;
  sherpa_onnx.OfflineTts? tts;
  String info = "Engine Active";

  @override
  void initState() {
    super.initState();
    // Initialize engines immediately as files are guaranteed to exist now
    _initEngines();
  }

  Future<void> _initEngines() async {
    try {
      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;

      // STT Config
      recognizer = sherpa_onnx.OnlineRecognizer(
        sherpa_onnx.OnlineRecognizerConfig(
          model: sherpa_onnx.OnlineModelConfig(
            transducer: sherpa_onnx.OnlineTransducerModelConfig(
              encoder: "$basePath/stt_root/tiny-encoder.int8.onnx",
              decoder: "$basePath/stt_root/tiny-decoder.int8.onnx",
              joiner: "$basePath/stt_root/tokens.txt",
            ),
            tokens: "$basePath/stt_root/tokens.txt",
            numThreads: 1,
          ),
        ),
      );

      // TTS Config
      tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
          model: sherpa_onnx.OfflineTtsModelConfig(
            vits: sherpa_onnx.OfflineTtsVitsModelConfig(
              model: "$basePath/tts_root/model.onnx",
              tokens: "$basePath/tts_root/tokens.txt",
              dataDir: "$basePath/tts_root/espeak-ng-data",
            ),
            provider: 'sherpa-onnx',
            numThreads: 1,
            debug: true,
          ),
        ),
      );

      setState(() => info = "Jarvis is Listening...");
      _speak("System online.");
    } catch (e) {
      setState(() => info = "Engine Init Error: $e");
    }
  }

  void _speak(String text) {
    if (tts != null) {
      tts!.generate(text: text, sid: 0, speed: 1.0);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("JARVIS CORE")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.graphic_eq, size: 100, color: Colors.blue),
            const SizedBox(height: 20),
            Text(info, style: const TextStyle(fontSize: 18)),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _speak("नमस्ते, मैं तैयार हूँ"),
              child: const Text("Test Voice"),
            )
          ],
        ),
      ),
    );
  }
}
