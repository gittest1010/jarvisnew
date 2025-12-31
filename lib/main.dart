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
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF1A1A1A),
        colorScheme: const ColorScheme.dark(primary: Colors.cyanAccent),
      ),
      home: const InitScreen(),
    );
  }
}

/* ================= INIT SCREEN ================= */

class InitScreen extends StatefulWidget {
  const InitScreen({super.key});
  @override
  State<InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<InitScreen> {
  String status = "Initializing System...";
  String logs = "";
  bool isError = false;
  Map<String, String> validPaths = {};

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

      // --- SMART FINDING (CRITICAL FIX) ---
      _log("Step 3: Searching for Model Files...");

      final sttDir = Directory("$basePath/stt_root");
      final ttsDir = Directory("$basePath/tts_root");

      // 1. Find STT Files (in stt-hi folder)
      final encoder = await _recursiveFind(sttDir, "tiny-encoder.int8.onnx");
      final decoder = await _recursiveFind(sttDir, "tiny-decoder.int8.onnx");

      // FIX: Check for both tokens.txt AND tokens.text
      var sttTokens = await _recursiveFind(sttDir, "tokens.txt");
      if (sttTokens == null) {
        _log("tokens.txt not found, checking tokens.text...");
        sttTokens = await _recursiveFind(sttDir, "tokens.text");
      }

      // 2. Find TTS Files (in tts-hi folder)
      final ttsModel = await _recursiveFind(ttsDir, "model.onnx");

      // FIX: Check for both tokens.txt AND tokens.text
      var ttsTokens = await _recursiveFind(ttsDir, "tokens.txt");
      if (ttsTokens == null) {
        ttsTokens = await _recursiveFind(ttsDir, "tokens.text");
      }

      final espeakData =
          await _recursiveFind(ttsDir, "espeak-ng-data", isFolder: true);

      // --- VALIDATION ---
      if (encoder == null) throw "STT Encoder missing (tiny-encoder.int8.onnx)";
      if (decoder == null) throw "STT Decoder missing (tiny-decoder.int8.onnx)";
      if (sttTokens == null) throw "STT Tokens missing (tokens.txt/text)";

      if (ttsModel == null) throw "TTS Model missing (model.onnx)";
      if (ttsTokens == null) throw "TTS Tokens missing (tokens.txt/text)";
      if (espeakData == null) throw "Espeak folder missing in tts_root";

      // Save paths
      validPaths = {
        "encoder": encoder,
        "decoder": decoder,
        "sttTokens": sttTokens,
        "ttsModel": ttsModel,
        "ttsTokens": ttsTokens,
        "espeakData": espeakData,
      };

      _log("All files located successfully!");
      await Future.delayed(const Duration(seconds: 1));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => JarvisHome(paths: validPaths)),
        );
      }
    } catch (e, stack) {
      _log("CRITICAL ERROR: $e", error: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  // --- SMART RECURSIVE SEARCH ---
  Future<String?> _recursiveFind(Directory dir, String filename,
      {bool isFolder = false}) async {
    try {
      if (!await dir.exists()) return null;

      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        // Check end of path to match filename regardless of parent folders
        if (entity.path.endsWith("/$filename") ||
            entity.path.endsWith("\\$filename")) {
          // Found it!
          if (isFolder && entity is Directory) {
            // Validate Espeak content
            if (filename == "espeak-ng-data") {
              if (File("${entity.path}/phontab").existsSync())
                return entity.path;
            } else {
              return entity.path;
            }
          } else if (!isFolder && entity is File) {
            if (entity.lengthSync() > 100)
              return entity.path; // Ignore empty files
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
    // Simple check: if folder exists and not empty, assume extracted
    if (await target.exists() && target.listSync().isNotEmpty) {
      _log("Skipping $folderName (Already exists)");
      return;
    }
    _log("Extracting $asset...");
    try {
      final data = await rootBundle.load(asset);
      final bytes = data.buffer.asUint8List();
      await compute(_backgroundUnzip, _UnzipArgs(bytes, basePath, folderName));
    } catch (e) {
      throw "Asset not found: $asset. Check pubspec.yaml";
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (!isError)
              const CircularProgressIndicator(color: Colors.cyanAccent),
            const SizedBox(height: 20),
            Text(status,
                textAlign: TextAlign.center,
                style: TextStyle(
                    color: isError ? Colors.red : Colors.white, fontSize: 16)),
            const SizedBox(height: 20),
            if (isError)
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(10),
                  color: Colors.black54,
                  child: SingleChildScrollView(
                      child: Text(logs,
                          style: const TextStyle(
                              fontFamily: 'monospace', fontSize: 12))),
                ),
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
    // Construct path carefully
    final filename = "${args.basePath}/${args.targetFolder}/${file.name}";
    if (file.isFile) {
      final f = File(filename);
      // Ensure directory exists
      if (!f.parent.existsSync()) f.parent.createSync(recursive: true);
      f.writeAsBytesSync(file.content as List<int>);
    } else {
      Directory(filename).createSync(recursive: true);
    }
  }
}

/* ================= JARVIS HOME ================= */

class JarvisHome extends StatefulWidget {
  final Map<String, String> paths;
  const JarvisHome({super.key, required this.paths});

  @override
  State<JarvisHome> createState() => _JarvisHomeState();
}

class _JarvisHomeState extends State<JarvisHome> {
  sherpa_onnx.OfflineTts? tts;
  String info = "Jarvis Active";

  @override
  void initState() {
    super.initState();
    _initEngine();
  }

  void _initEngine() {
    try {
      // Create TTS Engine with EXACT found paths
      final config = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: widget.paths["ttsModel"]!,
            tokens: widget.paths["ttsTokens"]!,
            dataDir: widget.paths["espeakData"]!,
          ),
          provider: 'sherpa-onnx',
          numThreads: 1,
        ),
      );
      tts = sherpa_onnx.OfflineTts(config);
      _speak("System Online");
    } catch (e) {
      setState(() => info = "Engine Error: $e");
    }
  }

  void _speak(String text) {
    tts?.generate(text: text, sid: 0, speed: 1.0);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Jarvis Core")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.mic, size: 80, color: Colors.cyanAccent),
            const SizedBox(height: 20),
            Text(info),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () => _speak("नमस्ते, मैं ठीक हूँ"),
              child: const Text("Test Voice"),
            )
          ],
        ),
      ),
    );
  }
}
