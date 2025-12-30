import 'dart:async';
import 'dart:io';

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
  String status = "Preparing models...";

  @override
  void initState() {
    super.initState();
    _prepare();
  }

  Future<void> _prepare() async {
    try {
      // 1. Initialize Native Bindings (CRITICAL)
      sherpa_onnx.initBindings();

      // 2. Extract Assets
      // Make sure these files exist in your pubspec.yaml
      await _extractOnce("assets/tts-hi.tar.bz2", "tts_hi");
      await _extractOnce("assets/stt-hi.tar.bz2", "stt_hi");

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const Home()),
        );
      }
    } catch (e) {
      setState(() => status = "Error: $e");
      debugPrint(e.toString());
    }
  }

  Future<void> _extractOnce(String asset, String folder) async {
    final base = await getApplicationDocumentsDirectory();
    final outDir = Directory("${base.path}/$folder");

    // Optimization: Check if directory exists and is not empty
    if (outDir.existsSync() && outDir.listSync().isNotEmpty) {
      debugPrint("Skipping extraction for $folder (already exists)");
      return;
    }

    setState(() => status = "Extracting $asset...");

    try {
      final data = await rootBundle.load(asset);
      final bytes = data.buffer.asUint8List();

      // Decode: BZip2 -> Tar
      final archive =
          TarDecoder().decodeBytes(BZip2Decoder().decodeBytes(bytes));

      for (final f in archive) {
        // Remove top-level folder from tar path if needed, or keep structure
        // This regex replaces the first directory in the path with our target folder
        // E.g. "espeak-ng-data/lang/..." -> "tts_hi/lang/..."
        final cleanName = f.name.replaceFirst(RegExp(r'^[^/]+'), folder);
        final outPath = "${base.path}/$cleanName";

        if (f.isFile) {
          final file = File(outPath);
          await file.parent.create(recursive: true);

          // Write bytes safely
          final content = f.content as List<int>;
          await file.writeAsBytes(content, flush: true);
        } else {
          await Directory(outPath).create(recursive: true);
        }
      }
    } catch (e) {
      debugPrint("Extraction failed for $asset: $e");
      // Continue execution, don't crash app (files might be manually placed)
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 20),
            Text(status, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}

/* ================= STT + TTS INIT ================= */

class Home extends StatefulWidget {
  const Home({super.key});
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  // Use OnlineRecognizer for 'tiny' models (Streaming)
  sherpa_onnx.OnlineRecognizer? recognizer;
  sherpa_onnx.OfflineTts? tts;

  String info = "Engines not initialized";

  @override
  void initState() {
    super.initState();
    _initEngines();
  }

  Future<void> _initEngines() async {
    final dir = await getApplicationDocumentsDirectory();

    try {
      // ✅ STT SETUP (Using Official Config)
      // Note: tiny-encoder/decoder are STREAMING models, so we use OnlineRecognizer
      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: "${dir.path}/stt_hi/tiny-encoder.int8.onnx",
            decoder: "${dir.path}/stt_hi/tiny-decoder.int8.onnx",
            joiner: "${dir.path}/stt_hi/tokens.txt",
          ),
          tokens: "${dir.path}/stt_hi/tokens.txt",
          numThreads: 1,
        ),
      );

      recognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      // ✅ TTS SETUP (Using Official Config)
      final ttsConfig = sherpa_onnx.OfflineTtsConfig(
        model: sherpa_onnx.OfflineTtsModelConfig(
          vits: sherpa_onnx.OfflineTtsVitsModelConfig(
            model: "${dir.path}/tts_hi/model.onnx",
            tokens: "${dir.path}/tts_hi/tokens.txt",
            dataDir: "${dir.path}/tts_hi/espeak-ng-data",
          ),
          provider: 'sherpa-onnx',
          numThreads: 1,
        ),
      );

      tts = sherpa_onnx.OfflineTts(ttsConfig);

      setState(() {
        info = "STT + TTS initialized successfully ✅";
      });
    } catch (e) {
      setState(() {
        info = "Initialization Error: $e";
      });
      debugPrint("Init Error: $e");
    }
  }

  void _testTts() {
    if (tts == null) return;

    try {
      final audio = tts!.generate(
          text: "नमस्ते, यह शेरपा ऑनक्स टीटीएस टेस्ट है", sid: 0, speed: 1.0);
      debugPrint("Generated samples: ${audio.samples.length}");
      setState(() {
        info = "TTS generated successfully (${audio.samples.length} samples)";
      });
      // Here you would use AudioPlayer to play the generated samples
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
      appBar: AppBar(title: const Text("Sherpa-ONNX Stable")),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Text(info, textAlign: TextAlign.center),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _testTts,
              child: const Text("Test TTS Generate"),
            ),
          ],
        ),
      ),
    );
  }
}
