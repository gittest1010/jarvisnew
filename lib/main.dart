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
import 'package:permission_handler/permission_handler.dart'; // Ensure this is imported

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
  String debugLog = "";

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
      // 1. Request Permissions
      _log("Requesting permissions...");
      Map<Permission, PermissionStatus> statuses = await [
        Permission.microphone,
      ].request();

      if (statuses[Permission.microphone] != PermissionStatus.granted) {
        _log("WARNING: Mic permission denied!");
      }

      // 2. Initialize Native Bindings
      _log("Loading native libraries...");
      sherpa_onnx.initBindings();

      // 3. Extract Assets in Background
      _log("Checking assets...");
      final baseDir = await getApplicationDocumentsDirectory();

      // Extract using SAFE logic
      await _extractAssetInBackground(
          "assets/tts-hi.tar.bz2", "tts_root", baseDir.path);

      await _extractAssetInBackground(
          "assets/stt-hi.tar.bz2", "stt_root", baseDir.path);

      _log("Assets ready. Starting Engine...");

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

    // Check if we need to extract
    if (await targetDir.exists()) {
      if (await targetDir.list().isEmpty) {
        _log("Re-extracting $targetFolder...");
      } else {
        _log("Skipping $targetFolder (Already exists).");
        return;
      }
    } else {
      await targetDir.create(recursive: true);
    }

    _log("Extracting $assetPath...");

    try {
      final ByteData data = await rootBundle.load(assetPath);
      final Uint8List bytes = data.buffer.asUint8List();

      // Run extraction in background
      await compute(
          _backgroundExtraction, _ExtractParams(bytes, basePath, targetFolder));

      _log("Extracted $targetFolder.");
    } catch (e) {
      throw Exception("Extraction failed for $assetPath: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(30.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!status.startsWith("CRITICAL"))
                const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Text(
                status,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color:
                      status.startsWith("CRITICAL") ? Colors.red : Colors.black,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (status.startsWith("CRITICAL")) ...[
                const SizedBox(height: 20),
                Expanded(
                  child: SingleChildScrollView(
                    child: Container(
                      padding: const EdgeInsets.all(10),
                      color: Colors.grey[200],
                      child: Text(debugLog,
                          style: const TextStyle(
                              fontSize: 10, fontFamily: 'monospace')),
                    ),
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

// --- BACKGROUND ISOLATE LOGIC (FIXED) ---
class _ExtractParams {
  final Uint8List bytes;
  final String basePath;
  final String targetFolder;
  _ExtractParams(this.bytes, this.basePath, this.targetFolder);
}

Future<void> _backgroundExtraction(_ExtractParams params) async {
  final archive =
      TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(params.bytes));

  for (final f in archive) {
    // FIX: सीधे targetFolder के अंदर डालो, नाम change मत करो।
    // इससे Flat Zip और Nested Zip दोनों काम करेंगे।
    final outPath = "${params.basePath}/${params.targetFolder}/${f.name}";

    if (f.isFile) {
      final file = File(outPath);
      if (!file.parent.existsSync()) file.parent.createSync(recursive: true);
      file.writeAsBytesSync(f.content as List<int>);
    } else {
      Directory(outPath).createSync(recursive: true);
    }
  }
}

/* ================= HOME & ENGINES ================= */

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

  // Helper to find file recursively
  Future<String?> _findPath(Directory dir, String filename,
      {bool isFolder = false}) async {
    try {
      if (!await dir.exists()) return null;
      final entities = await dir.list(recursive: true).toList();
      for (var entity in entities) {
        if (entity.path.endsWith(filename)) {
          if (isFolder && entity is Directory) return entity.path;
          if (!isFolder && entity is File) return entity.path;
        }
      }
    } catch (e) {
      print("Error finding file: $e");
    }
    return null;
  }

  Future<void> _initEngines() async {
    final docDir = await getApplicationDocumentsDirectory();
    final sttRoot = Directory("${docDir.path}/stt_root");
    final ttsRoot = Directory("${docDir.path}/tts_root");

    try {
      // 1. SMART FIND: Locate files wherever they are
      final encoder = await _findPath(sttRoot, "tiny-encoder.int8.onnx");
      final decoder = await _findPath(sttRoot, "tiny-decoder.int8.onnx");
      final tokensSTT = await _findPath(sttRoot, "tokens.txt");

      final modelTTS = await _findPath(ttsRoot, "model.onnx");
      final tokensTTS = await _findPath(ttsRoot, "tokens.txt");
      // Find 'espeak-ng-data' folder
      final espeakData =
          await _findPath(ttsRoot, "espeak-ng-data", isFolder: true);

      if (encoder == null || decoder == null || tokensSTT == null) {
        throw Exception(
            "STT Model files missing inside ${sttRoot.path}. Did extraction work?");
      }
      if (modelTTS == null || tokensTTS == null || espeakData == null) {
        throw Exception(
            "TTS Model files missing inside ${ttsRoot.path}. Did extraction work?");
      }

      // 2. Initialize STT
      recognizer = sherpa_onnx.OnlineRecognizer(
        sherpa_onnx.OnlineRecognizerConfig(
          model: sherpa_onnx.OnlineModelConfig(
            transducer: sherpa_onnx.OnlineTransducerModelConfig(
              encoder: encoder,
              decoder: decoder,
              joiner: tokensSTT,
            ),
            tokens: tokensSTT,
            numThreads: 1,
          ),
        ),
      );

      // 3. Initialize TTS
      tts = sherpa_onnx.OfflineTts(
        sherpa_onnx.OfflineTtsConfig(
          model: sherpa_onnx.OfflineTtsModelConfig(
            vits: sherpa_onnx.OfflineTtsVitsModelConfig(
              model: modelTTS,
              tokens: tokensTTS,
              dataDir: espeakData,
            ),
            provider: 'sherpa-onnx',
            numThreads: 1,
          ),
        ),
      );

      setState(() => info = "Jarvis Ready 🟢\nTap below to test.");
    } catch (e) {
      setState(() => info = "Engine Init Failed:\n$e");
      debugPrint("Init Error: $e");
    }
  }

  void _testTts() {
    if (tts == null) return;
    try {
      final audio =
          tts!.generate(text: "नमस्ते, सब ठीक है", sid: 0, speed: 1.0);
      setState(() => info = "Generated ${audio.samples.length} samples");
    } catch (e) {
      setState(() => info = "TTS Error: $e");
    }
  }

  @override
  void dispose() {
    recognizer?.free();
    tts?.free();
    super.dispose();
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
              child: Text(info, textAlign: TextAlign.center),
            ),
            ElevatedButton(
              onPressed: _testTts,
              child: const Text("Test TTS (नमस्ते)"),
            ),
          ],
        ),
      ),
    );
  }
}
