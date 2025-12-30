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
import 'package:permission_handler/permission_handler.dart';

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
  bool hasError = false;

  @override
  void initState() {
    super.initState();
    _startInitialization();
  }

  void _log(String message, {bool isError = false}) {
    debugPrint(message);
    if (mounted) {
      setState(() {
        status = message;
        if (isError) hasError = true;
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
        Permission.storage, // Good to request explicit storage too
      ].request();

      if (statuses[Permission.microphone] != PermissionStatus.granted) {
        _log("WARNING: Mic permission denied!", isError: true);
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

      // Short delay to ensure UI updates before navigation
      await Future.delayed(const Duration(milliseconds: 500));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const Home()),
        );
      }
    } catch (e, stack) {
      _log("CRITICAL ERROR: $e", isError: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  Future<void> _extractAssetInBackground(
      String assetPath, String targetFolder, String basePath) async {
    final targetDir = Directory("$basePath/$targetFolder");

    // Check if we need to extract (Simple check: folder exists and not empty)
    if (await targetDir.exists()) {
      if (await targetDir.list().isEmpty) {
        _log("Re-extracting $targetFolder (Empty)...");
      } else {
        _log("Skipping $targetFolder (Already exists).");
        return;
      }
    } else {
      await targetDir.create(recursive: true);
    }

    _log("Extracting $assetPath...");

    try {
      // Load asset data
      final ByteData data = await rootBundle.load(assetPath);
      final Uint8List bytes = data.buffer.asUint8List();

      // Run extraction in background isolate
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
              if (!hasError) const CircularProgressIndicator(),
              const SizedBox(height: 20),
              Text(
                status,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: hasError ? Colors.red : Colors.black,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (hasError || debugLog.length > 200) ...[
                const SizedBox(height: 20),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(10),
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Colors.grey[200],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: SingleChildScrollView(
                      child: Text(
                        debugLog,
                        style: const TextStyle(
                            fontSize: 10, fontFamily: 'monospace'),
                      ),
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

// --- BACKGROUND ISOLATE LOGIC ---
class _ExtractParams {
  final Uint8List bytes;
  final String basePath;
  final String targetFolder;
  _ExtractParams(this.bytes, this.basePath, this.targetFolder);
}

Future<void> _backgroundExtraction(_ExtractParams params) async {
  // Decode BZip2 -> Tar
  final archive =
      TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(params.bytes));

  for (final f in archive) {
    // Construct safe output path
    // Remove potentially unsafe characters from filename if necessary
    final safeName = f.name.replaceAll("../", "");
    final outPath = "${params.basePath}/${params.targetFolder}/$safeName";

    if (f.isFile) {
      final file = File(outPath);
      // Ensure parent directory exists
      if (!file.parent.existsSync()) {
        file.parent.createSync(recursive: true);
      }
      // Write bytes
      file.writeAsBytesSync(f.content as List<int>);
    } else {
      // It's a directory
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
  bool isReady = false;

  @override
  void initState() {
    super.initState();
    // Run after build to prevent UI freeze during initial frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initEngines();
    });
  }

  // Helper to find file recursively and return absolute path
  Future<String?> _findPath(Directory dir, String filename,
      {bool isFolder = false}) async {
    try {
      if (!await dir.exists()) return null;

      final entities = await dir.list(recursive: true).toList();
      for (var entity in entities) {
        // Match filename strictly
        if (entity.path.endsWith(Platform.pathSeparator + filename) ||
            entity.path.endsWith("/$filename")) {
          if (isFolder && entity is Directory) {
            // CRITICAL CHECK for Espeak: It must contain 'phontab'
            if (filename == "espeak-ng-data") {
              final phontab = File("${entity.path}/phontab");
              if (!phontab.existsSync()) {
                print(
                    "Found espeak folder but 'phontab' is missing: ${entity.path}");
                continue; // Keep looking
              }
            }
            return entity.path;
          }

          if (!isFolder && entity is File) {
            // CRITICAL CHECK: File must not be empty
            if (await entity.length() > 0) {
              return entity.path;
            } else {
              print("Found $filename but it is empty (0 bytes).");
            }
          }
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

    String log = "";
    bool sttOk = false;
    bool ttsOk = false;

    // --- 1. Initialize STT ---
    try {
      final encoder = await _findPath(sttRoot, "tiny-encoder.int8.onnx");
      final decoder = await _findPath(sttRoot, "tiny-decoder.int8.onnx");
      final tokensSTT = await _findPath(sttRoot, "tokens.txt");

      if (encoder == null || decoder == null || tokensSTT == null) {
        log += "❌ STT Files Missing.\n";
      } else {
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
        log += "✅ STT Ready.\n";
        sttOk = true;
      }
    } catch (e) {
      log += "❌ STT Crash: $e\n";
      print("STT Error: $e");
    }

    // --- 2. Initialize TTS ---
    try {
      final modelTTS = await _findPath(ttsRoot, "model.onnx");
      final tokensTTS = await _findPath(ttsRoot, "tokens.txt");
      final espeakData =
          await _findPath(ttsRoot, "espeak-ng-data", isFolder: true);

      if (modelTTS == null || tokensTTS == null) {
        log += "❌ TTS Files Missing.\n";
      } else if (espeakData == null) {
        log += "❌ 'espeak-ng-data' folder missing or invalid.\n";
      } else {
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
              debug: true, // Enable debug logs from C++
            ),
          ),
        );
        log += "✅ TTS Ready.\n";
        ttsOk = true;
      }
    } catch (e) {
      log += "❌ TTS Crash: $e\n";
      print("TTS Error: $e");
    }

    if (mounted) {
      setState(() {
        info = log;
        isReady = sttOk || ttsOk;
      });
    }
  }

  void _testTts() {
    if (tts == null) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("TTS Engine not loaded!")));
      return;
    }
    try {
      // Basic text generation test
      final audio =
          tts!.generate(text: "नमस्ते, आप कैसे हैं?", sid: 0, speed: 1.0);

      setState(() => info += "\n🔊 Generated ${audio.samples.length} samples");

      // Note: Actual audio playback needs the 'audioplayers' package or similar.
      // This just proves the engine generated raw audio data.
    } catch (e) {
      setState(() => info += "\n❌ Gen Error: $e");
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
      appBar: AppBar(title: const Text("Sherpa Safe Mode")),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  info,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontFamily: "monospace"),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: isReady ? _testTts : null,
                icon: const Icon(Icons.volume_up),
                label: const Text("Test TTS (नमस्ते)"),
              ),
              const SizedBox(height: 10),
              if (!isReady) const CircularProgressIndicator()
            ],
          ),
        ),
      ),
    );
  }
}
