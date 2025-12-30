import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart'; // Required for compute
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
// Archive v4 compatible
import 'package:archive/archive.dart';
import 'package:archive/archive_io.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: InitScreen(),
    );
  }
}

/* ================= INIT & EXTRACT ================= */

class InitScreen extends StatefulWidget {
  const InitScreen({super.key});
  @override
  State<InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<InitScreen> {
  String status = "Initializing...";
  String debugLog = ""; // For showing detailed errors

  @override
  void initState() {
    super.initState();
    _startInitialization();
  }

  void _log(String message) {
    debugPrint(message);
    if (mounted) {
      setState(() {
        status = message;
        debugLog += "\n$message";
      });
    }
  }

  Future<void> _startInitialization() async {
    try {
      // 1. Initialize Native Bindings
      _log("Loading native libraries...");
      sherpa_onnx.initBindings();
      _log("Native libraries loaded.");

      // 2. Extract Assets in Background
      _log("Checking assets...");
      final baseDir = await getApplicationDocumentsDirectory();

      // Extract TTS Data
      await _extractAssetInBackground(
          "assets/tts-hi.tar.bz2", "tts_hi", baseDir.path);

      // Extract STT Data
      await _extractAssetInBackground(
          "assets/stt-hi.tar.bz2", "stt_hi", baseDir.path);

      _log("All assets ready. Launching App...");

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const Home()),
        );
      }
    } catch (e, stack) {
      _log("CRITICAL ERROR: $e");
      debugPrintStack(stackTrace: stack);
    }
  }

  Future<void> _extractAssetInBackground(
      String assetPath, String targetFolder, String basePath) async {
    final targetDir = Directory("$basePath/$targetFolder");

    // Optimization: Check if critical files exist to avoid re-extraction
    // We assume if the folder exists and has content, it's good.
    if (await targetDir.exists()) {
      if (await targetDir.list().isEmpty) {
        _log("Folder $targetFolder is empty. Re-extracting...");
      } else {
        _log("Skipping $targetFolder (Already exists).");
        return;
      }
    }

    _log("Extracting $assetPath...");

    try {
      // 1. Load asset bytes (Must be done on Main Thread)
      final ByteData data = await rootBundle.load(assetPath);
      final Uint8List bytes = data.buffer.asUint8List();

      // 2. Pass to Background Isolate for Heavy Lifting
      await compute(
          _backgroundExtraction, _ExtractParams(bytes, basePath, targetFolder));

      _log("Extracted $targetFolder successfully.");
    } catch (e) {
      if (e is FlutterError && e.message.contains("Unable to load asset")) {
        throw Exception(
            "ASSET MISSING: Could not find '$assetPath'. Check pubspec.yaml!");
      }
      throw Exception("Extraction failed for $assetPath: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!status.startsWith("CRITICAL ERROR"))
                const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Text(
                status,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color:
                      status.startsWith("CRITICAL") ? Colors.red : Colors.black,
                  fontWeight: status.startsWith("CRITICAL")
                      ? FontWeight.bold
                      : FontWeight.normal,
                ),
              ),
              if (status.startsWith("CRITICAL")) ...[
                const SizedBox(height: 20),
                const Text("Debug Log:",
                    style: TextStyle(fontWeight: FontWeight.bold)),
                Expanded(
                  child: SingleChildScrollView(
                    child: Text(debugLog,
                        style: const TextStyle(
                            fontSize: 10, fontFamily: 'monospace')),
                  ),
                ),
              ]
            ],
          ),
        ),
      ),
    );
  }
}

// --- BACKGROUND ISOLATE LOGIC ---

class _ExtractParams {
  final Uint8List bytes;
  final String basePath;
  final String targetFolder;

  _ExtractParams(this.bytes, this.basePath, this.targetFolder);
}

// This function runs in a separate thread (Isolate)
Future<void> _backgroundExtraction(_ExtractParams params) async {
  try {
    // Decode BZip2 -> Tar
    final archive =
        TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(params.bytes));

    for (final f in archive) {
      // Fix path: Replace root folder name in tar with our target folder name
      // e.g. "espeak-ng-data/lang" -> "tts_hi/lang"
      final cleanName =
          f.name.replaceFirst(RegExp(r'^[^/]+'), params.targetFolder);
      final outPath = "${params.basePath}/$cleanName";

      if (f.isFile) {
        final file = File(outPath);
        if (!file.parent.existsSync()) {
          file.parent.createSync(recursive: true);
        }
        file.writeAsBytesSync(f.content as List<int>);
      } else {
        final dir = Directory(outPath);
        if (!dir.existsSync()) {
          dir.createSync(recursive: true);
        }
      }
    }
  } catch (e) {
    print("Background extraction error: $e");
    throw e;
  }
}

/* ================= STT + TTS INIT ================= */

class Home extends StatefulWidget {
  const Home({super.key});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  sherpa_onnx.OnlineRecognizer? recognizer;
  sherpa_onnx.OfflineTts? tts;
  String info = "Initializing engines...";

  @override
  void initState() {
    super.initState();
    _initEngines();
  }

  Future<void> _initEngines() async {
    final dir = await getApplicationDocumentsDirectory();
    final sttPath = "${dir.path}/stt_hi";
    final ttsPath = "${dir.path}/tts_hi";

    try {
      // ✅ Verify Files Exist Before Init
      _checkFile(sttPath, "tiny-encoder.int8.onnx");
      _checkFile(sttPath, "tiny-decoder.int8.onnx");
      _checkFile(sttPath, "tokens.txt");
      _checkFile(ttsPath, "model.onnx");
      _checkFile(ttsPath, "espeak-ng-data"); // Folder check

      // ✅ STT SETUP
      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: "$sttPath/tiny-encoder.int8.onnx",
            decoder: "$sttPath/tiny-decoder.int8.onnx",
            joiner: "$sttPath/tokens.txt",
          ),
          tokens: "$sttPath/tokens.txt",
          numThreads: 1,
        ),
      );
      recognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      // ✅ TTS SETUP
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: "$ttsPath/model.onnx",
            tokens: "$ttsPath/tokens.txt",
            dataDir: "$ttsPath/espeak-ng-data",
          ),
          provider: 'sherpa-onnx',
          numThreads: 1,
        ),
      );
      tts = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() => info = "Jarvis Ready 🟢");
    } catch (e) {
      setState(() => info = "Init Failed: $e");
    }
  }

  void _checkFile(String basePath, String filename) {
    final path = "$basePath/$filename";
    final type = filename.contains(".") ? "File" : "Folder"; // Simple check
    if (type == "File" && !File(path).existsSync()) {
      throw Exception("MISSING FILE: $filename at $basePath");
    } else if (type == "Folder" && !Directory(path).existsSync()) {
      throw Exception("MISSING FOLDER: $filename at $basePath");
    }
  }

  void _testTts() {
    if (tts == null) return;
    try {
      final audio =
          tts!.generate(text: "नमस्ते, टेस्ट सफल रहा", sid: 0, speed: 1.0);
      setState(() => info = "TTS Generated: ${audio.samples.length} samples");
    } catch (e) {
      setState(() => info = "TTS Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Jarvis AI")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(info,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 16)),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _testTts,
              child: const Text("Test Speak (नमस्ते)"),
            ),
          ],
        ),
      ),
    );
  }
}
