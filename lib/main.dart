import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_sound/flutter_sound.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(home: InitScreen());
  }
}

/* ================= INIT & MODEL EXTRACT ================= */

class InitScreen extends StatefulWidget {
  const InitScreen({super.key});
  @override
  State<InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<InitScreen> {
  String status = "Initializing…";

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    await Permission.microphone.request();

    await _extractOnce("assets/tts-hi.tar.bz2", "tts_hi");
    await _extractOnce("assets/stt-hi.tar.bz2", "stt_hi");

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const EngineScreen()),
    );
  }

  Future<void> _extractOnce(String asset, String folder) async {
    final base = await getApplicationDocumentsDirectory();
    final outDir = Directory("${base.path}/$folder");
    if (outDir.existsSync()) return;

    final data = await rootBundle.load(asset);
    final bytes = data.buffer.asUint8List();
    final tar = TarDecoder().decodeBytes(
      BZip2Decoder().decodeBytes(bytes),
    );

    for (final f in tar) {
      final outPath =
          "${base.path}/${f.name.replaceFirst(RegExp(r'^[^/]+'), folder)}";
      if (f.isFile) {
        File(outPath)
          ..createSync(recursive: true)
          ..writeAsBytesSync(f.content as List<int>);
      } else {
        Directory(outPath).createSync(recursive: true);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(child: Text(status)),
    );
  }
}

/* ================= MIC → STT → TTS ================= */

class EngineScreen extends StatefulWidget {
  const EngineScreen({super.key});
  @override
  State<EngineScreen> createState() => _EngineScreenState();
}

class _EngineScreenState extends State<EngineScreen> {
  late OfflineRecognizer recognizer;
  late OfflineTts tts;
  late FlutterSoundRecorder recorder;

  bool running = false;

  @override
  void initState() {
    super.initState();
    _initEngines();
  }

  Future<void> _initEngines() async {
    final dir = await getApplicationDocumentsDirectory();

    recognizer = OfflineRecognizer(
      OfflineRecognizerConfig(
        model: OfflineRecognizerModelConfig(
          transducer: OfflineRecognizerTransducerModelConfig(
            encoder: "${dir.path}/stt_hi/tiny-encoder.int8.onnx",
            decoder: "${dir.path}/stt_hi/tiny-decoder.int8.onnx",
            tokens: "${dir.path}/stt_hi/tokens.txt",
          ),
        ),
      ),
    );

    tts = OfflineTts(
      OfflineTtsConfig(
        model: OfflineTtsModelConfig(
          vits: OfflineTtsVitsModelConfig(
            model: "${dir.path}/tts_hi/model.onnx",
            tokens: "${dir.path}/tts_hi/tokens.txt",
            dataDir: "${dir.path}/tts_hi/espeak-ng-data",
          ),
        ),
      ),
    );

    recorder = FlutterSoundRecorder();
    await recorder.openRecorder();
  }

  Future<void> _startListening() async {
    if (running) return;
    running = true;

    final stream = StreamController<Uint8List>();

    await recorder.startRecorder(
      codec: Codec.pcm16,
      sampleRate: 16000,
      numChannels: 1,
      toStream: stream.sink,
    );

    stream.stream.listen((data) {
      recognizer.acceptWaveform(data);
      if (recognizer.isReady) {
        final result = recognizer.decode();
        if (result.text.isNotEmpty) {
          final audio = tts.generate(result.text);
          audio.play();
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Mic STT → TTS")),
      body: Center(
        child: ElevatedButton(
          onPressed: _startListening,
          child: const Text("START LISTENING"),
        ),
      ),
    );
  }
}
