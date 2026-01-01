import 'dart:async';
import 'dart:io' as io;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

// Archive
import 'package:archive/archive.dart';
// archive_io import optional tha, hata diya kyunki use nahi ho raha
// import 'package:archive/archive_io.dart';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';

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
        scaffoldBackgroundColor: const Color(0xFF0A0E27),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF00E5FF),
          secondary: Color(0xFFFF1744),
        ),
      ),
      home: const InitScreen(),
    );
  }
}

enum SttModelType { transducer, whisper }

/* ==================== INIT SCREEN (SMART SETUP) ==================== */
class InitScreen extends StatefulWidget {
  const InitScreen({super.key});
  @override
  State<InitScreen> createState() => _InitScreenState();
}

class _InitScreenState extends State<InitScreen> {
  String status = "Initializing Jarvis...";
  String logs = "";
  bool isError = false;
  Map<String, String> validPaths = {};
  SttModelType detectedModelType = SttModelType.whisper; // Default to Whisper

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
      _log("🔐 Checking Permissions...");
      await Permission.microphone.request();
      await Permission.storage.request();

      _log("⚙️ Init Sherpa Bindings...");
      try {
        sherpa.initBindings();
      } catch (e) {
        debugPrint("Bindings info: $e");
      }

      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;
      final sttDir = io.Directory("$basePath/stt_root");
      final ttsDir = io.Directory("$basePath/tts_root");

      // Extraction
      await _extractIfNeeded("assets/stt-hi.tar.bz2", "stt_root", basePath);
      await _extractIfNeeded("assets/tts-hi.tar.bz2", "tts_root", basePath);

      // Detecting Model Architecture
      _log("🔍 Detecting Model Architecture...");

      final String? encoder = await _smartFind(sttDir, [
        "tiny-encoder.onnx",
        "tiny-encoder.int8.onnx",
        "encoder.onnx",
        "encoder.int8.onnx",
        "base-encoder.onnx",
        "small-encoder.onnx",
      ]);

      final String? decoder = await _smartFind(sttDir, [
        "tiny-decoder.onnx",
        "tiny-decoder.int8.onnx",
        "decoder.onnx",
        "decoder.int8.onnx",
        "base-decoder.onnx",
        "small-decoder.onnx",
      ]);

      // Whisper does NOT use a joiner. We only look for it if we suspect Transducer.
      // But user explicitly wants Whisper Tiny, so we prioritize that.
      final String? joiner = await _smartFind(sttDir, [
        "joiner.onnx",
        "joiner.int8.onnx",
        "joiner-epoch-99-avg-1.onnx",
      ]);

      final String? sttTokens = await _smartFind(sttDir, ["tokens.txt"]);

      // Decide model type
      // Decide model type
      if (encoder != null && decoder != null && sttTokens != null) {
         // Whisper detection (Priority)
         _log("✅ Whisper Model Detected.");
         detectedModelType = SttModelType.whisper;
      } else if (joiner != null && encoder != null && decoder != null && sttTokens != null) {
          // Transducer detection
          _log("✅ Transducer Model Detected.");
          detectedModelType = SttModelType.transducer;
      } else {
         // Detailed error reporting
         String missing = "";
         if (encoder == null) missing += "Encoder, ";
         if (decoder == null) missing += "Decoder, ";
         if (sttTokens == null) missing += "Tokens, ";
         if (joiner == null && encoder != null && decoder != null) missing += "(Joiner - ignored for Whisper), ";
         
         throw "❌ Invalid Model Files.\nMissing: $missing\nFound: Encoder=${encoder != null}, Decoder=${decoder != null}, Tokens=${sttTokens != null}, Joiner=${joiner != null}";
      }

      // TTS Files
      final ttsModel =
          await _smartFind(ttsDir, ["model.onnx", "vits-model.onnx"]);
      final ttsTokens = await _smartFind(ttsDir, ["tokens.txt"]);
      final espeakData = await _smartFindFolder(ttsDir, "espeak-ng-data");

      // Validation
      if (encoder == null) throw "STT Encoder Missing!";
      if (decoder == null) throw "STT Decoder Missing!";
      if (sttTokens == null) throw "STT Tokens Missing!";
      if (ttsModel == null) throw "TTS Model Missing!";
      if (ttsTokens == null) throw "TTS Tokens Missing!";
      if (espeakData == null) throw "eSpeak Data Missing!";

      validPaths = {
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner ?? "", // Optional for Whisper
        "sttTokens": sttTokens,
        "ttsModel": ttsModel,
        "ttsTokens": ttsTokens,
        "espeakData": espeakData,
      };

      _log("🚀 Ready! Mode: ${detectedModelType.name.toUpperCase()}");
      await Future.delayed(const Duration(seconds: 1));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => JarvisHome(
              paths: validPaths,
              modelType: detectedModelType,
            ),
          ),
        );
      }
    } catch (e, stack) {
      _log("💥 ERROR: $e", error: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  Future<String?> _smartFind(io.Directory dir, List<String> patterns) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity is io.File) {
          final name = entity.path.split('/').last.toLowerCase();
          for (var p in patterns) {
            if (name == p.toLowerCase() || name.contains(p.toLowerCase())) {
              return entity.path;
            }
          }
        }
      }
    } catch (_) {}
    return null;
  }

  Future<String?> _smartFindFolder(io.Directory dir, String folderName) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity is io.Directory) {
          if (entity.path.split('/').last == folderName) return entity.path;
        }
      }
    } catch (_) {}
    return null;
  }

  Future<void> _extractIfNeeded(
      String asset, String folderName, String basePath) async {
    final target = io.Directory("$basePath/$folderName");
    if (await target.exists() && target.listSync().isNotEmpty) {
      _log("✓ $folderName ready");
      return;
    }
    _log("📦 Extracting $folderName...");
    try {
      final data = await rootBundle.load(asset);
      final bytes = data.buffer.asUint8List();

      // compute ko isolate-safe banaya: custom class ki jagah Map pass kiya
      await compute(_backgroundUnzip, {
        "bytes": bytes,
        "basePath": basePath,
        "targetFolder": folderName,
      });
    } catch (e) {
      _log("⚠️ Extract Error: $e (Check assets)");
      rethrow;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        padding: const EdgeInsets.all(20),
        decoration: const BoxDecoration(
          gradient:
              LinearGradient(colors: [Color(0xFF0A0E27), Color(0xFF1A1A2E)]),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (!isError)
              const CircularProgressIndicator(color: Color(0xFF00E5FF)),
            const SizedBox(height: 20),
            Text(
              status,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: isError ? Colors.redAccent : const Color(0xFF00E5FF),
              ),
            ),
            if (isError) ...[
              const SizedBox(height: 20),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(10),
                  color: Colors.black54,
                  child: SingleChildScrollView(
                    child: Text(
                      logs,
                      style: const TextStyle(
                          fontFamily: 'monospace', fontSize: 11),
                    ),
                  ),
                ),
              )
            ]
          ],
        ),
      ),
    );
  }
}

Future<void> _backgroundUnzip(Map<String, Object> args) async {
  final bytes = args["bytes"] as Uint8List;
  final basePath = args["basePath"] as String;
  final targetFolder = args["targetFolder"] as String;

  final List<int> bzip2Bytes = BZip2Decoder().decodeBytes(bytes);
  final archive = TarDecoder().decodeBytes(bzip2Bytes);

  for (final file in archive) {
    final filename = "$basePath/$targetFolder/${file.name}";
    if (file.isFile) {
      final f = io.File(filename);
      if (!f.parent.existsSync()) f.parent.createSync(recursive: true);
      f.writeAsBytesSync(file.content as List<int>);
    } else {
      io.Directory(filename).createSync(recursive: true);
    }
  }
}

/* ==================== JARVIS INTERFACE (DUAL ENGINE) ==================== */
class JarvisHome extends StatefulWidget {
  final Map<String, String> paths;
  final SttModelType modelType;

  const JarvisHome({super.key, required this.paths, required this.modelType});

  @override
  State<JarvisHome> createState() => _JarvisHomeState();
}

class _JarvisHomeState extends State<JarvisHome>
    with SingleTickerProviderStateMixin {
  // TTS
  sherpa.OfflineTts? _tts;

  // STT Engines
  sherpa.OnlineRecognizer? _transducerRecognizer;
  sherpa.OnlineStream? _transducerStream;

  sherpa.OfflineRecognizer? _whisperRecognizer;
  sherpa.OfflineStream? _whisperStream;

  // Active engine
  SttModelType? _activeModelType;

  // Audio Recorder (record v5.x)
  final Record _recorder = Record();
  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription<Uint8List>? _audioSub;

  bool _isListening = false;
  bool _isSpeaking = false;
  bool _isProcessing = false;
  String _transcribedText = "";
  String _statusMessage = "Ready";

  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 0.8, end: 1.2).animate(
        CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut));

    _initAI();
  }

  void _initAI() async {
    try {
      _log("🧠 Starting AI (${widget.modelType.name})...");

      // TTS
      if (widget.paths["ttsModel"] == null ||
          widget.paths["espeakData"] == null) {
        throw "TTS Model or Config Missing";
      }

      _tts = sherpa.OfflineTts(
        sherpa.OfflineTtsConfig(
          model: sherpa.OfflineTtsModelConfig(
            vits: sherpa.OfflineTtsVitsModelConfig(
              model: widget.paths["ttsModel"]!,
              tokens: widget.paths["ttsTokens"]!,
              dataDir: widget.paths["espeakData"]!,
            ),
            numThreads: 1,
            debug: false,
          ),
        ),
      );

      // STT Initialization
      // STT Initialization
      if (widget.modelType == SttModelType.whisper) {
        _log("🔹 Initializing Whisper...");
        _whisperRecognizer = sherpa.OfflineRecognizer(
          sherpa.OfflineRecognizerConfig(
            model: sherpa.OfflineModelConfig(
              whisper: sherpa.OfflineWhisperModelConfig(
                encoder: widget.paths["encoder"]!,
                decoder: widget.paths["decoder"]!,
              ),
              tokens: widget.paths["sttTokens"]!,
              numThreads: 1,
              debug: true,
            ),
          ),
        );
        _activeModelType = SttModelType.whisper;
      } else {
        // Transducer fallback (only if joiner exists)
        final joinerPath = widget.paths["joiner"];
        if (joinerPath != null && joinerPath.isNotEmpty) {
           _log("🔹 Initializing Transducer...");
           _transducerRecognizer = sherpa.OnlineRecognizer(
            sherpa.OnlineRecognizerConfig(
              model: sherpa.OnlineModelConfig(
                transducer: sherpa.OnlineTransducerModelConfig(
                  encoder: widget.paths["encoder"]!,
                  decoder: widget.paths["decoder"]!,
                  joiner: joinerPath,
                ),
                tokens: widget.paths["sttTokens"]!,
                numThreads: 1,
                debug: true,
              ),
              enableEndpoint: true,
            ),
          );
          _activeModelType = SttModelType.transducer;
        } else {
           throw "Transducer model selected but joiner not found! (Path is empty)";
        }
      }

      _log("✅ Online");
      _speak(
          "System ready. Using ${_activeModelType?.name ?? widget.modelType.name} model.");
    } catch (e) {
      _log("❌ Init Error: $e");
      debugPrint("Full Error: $e");
    }
  }

  void _log(String msg) {
    if (mounted) setState(() => _statusMessage = msg);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _audioSub?.cancel();
    _recorder.dispose();
    _transducerStream?.free();
    _transducerRecognizer?.free();
    _whisperStream?.free();
    _whisperRecognizer?.free();
    _tts?.free();
    _audioPlayer.dispose();
    super.dispose();
  }

  Future<void> _speak(String text) async {
    if (text.isEmpty || _tts == null) return;
    setState(() {
      _isSpeaking = true;
      _statusMessage = "🔊 Speaking...";
    });
    try {
      await _audioPlayer.stop();

      final audio = _tts!.generate(text: text, sid: 0, speed: 1.0);
      final pcm = Int16List(audio.samples.length);
      for (int i = 0; i < audio.samples.length; i++) {
        final s = (audio.samples[i] * 32767.0).round();
        pcm[i] = s.clamp(-32768, 32767);
      }
      final tempDir = await getTemporaryDirectory();
      final wavPath = "${tempDir.path}/jarvis_out.wav";

      io.File(wavPath).writeAsBytesSync(_createWav(pcm, audio.sampleRate));
      await _audioPlayer.play(DeviceFileSource(wavPath));
      await _audioPlayer.onPlayerComplete.first;
    } catch (_) {}
    if (mounted) setState(() => _isSpeaking = false);
  }

  Uint8List _createWav(Int16List pcm, int sampleRate) {
    final bytesBuilder = BytesBuilder();

    bytesBuilder.add(Uint8List.fromList("RIFF".codeUnits));
    bytesBuilder.add(_int32Bytes(36 + pcm.lengthInBytes));

    bytesBuilder.add(Uint8List.fromList("WAVE".codeUnits));

    bytesBuilder.add(Uint8List.fromList("fmt ".codeUnits));
    bytesBuilder.add(_int32Bytes(16));
    bytesBuilder.add(_int16Bytes(1));
    bytesBuilder.add(_int16Bytes(1));
    bytesBuilder.add(_int32Bytes(sampleRate));
    bytesBuilder.add(_int32Bytes(sampleRate * 2));
    bytesBuilder.add(_int16Bytes(2));
    bytesBuilder.add(_int16Bytes(16));

    bytesBuilder.add(Uint8List.fromList("data".codeUnits));
    bytesBuilder.add(_int32Bytes(pcm.lengthInBytes));
    bytesBuilder.add(pcm.buffer.asUint8List());

    return bytesBuilder.toBytes();
  }

  Uint8List _int32Bytes(int v) =>
      Uint8List(4)..buffer.asByteData().setInt32(0, v, Endian.little);
  Uint8List _int16Bytes(int v) =>
      Uint8List(2)..buffer.asByteData().setInt16(0, v, Endian.little);

  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    final sttType = _activeModelType ?? widget.modelType;

    if ((_transducerRecognizer == null && _whisperRecognizer == null) ||
        _isSpeaking) return;
    if (!await _recorder.hasPermission()) return;

    if (sttType == SttModelType.transducer) {
      if (_transducerRecognizer != null) {
        _transducerStream = _transducerRecognizer!.createStream();
      }
    } else {
      if (_whisperRecognizer != null) {
        _whisperStream = _whisperRecognizer!.createStream();
      }
    }

    try {
      // startStream returns Future<Stream<Uint8List>> => await required
      final stream = await _recorder.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 16000,
          numChannels: 1,
        ),
      );

      setState(() {
        _isListening = true;
        _transcribedText = "";
        _statusMessage = sttType == SttModelType.whisper
            ? "🎤 Listening (Processing at end)..."
            : "🎤 Listening...";
      });

      _audioSub = stream.listen((data) {
        if (data.isEmpty) return;
        if (data.lengthInBytes % 2 != 0) return;

        final int16s = Int16List.view(
          data.buffer,
          data.offsetInBytes,
          data.lengthInBytes ~/ 2,
        );
        final float32s = Float32List(int16s.length);
        for (int i = 0; i < int16s.length; i++) {
          float32s[i] = int16s[i] / 32768.0;
        }

        if (sttType == SttModelType.transducer && _transducerStream != null) {
          _transducerStream!.acceptWaveform(
            samples: float32s,
            sampleRate: 16000,
          );

          while (_transducerRecognizer!.isReady(_transducerStream!)) {
            _transducerRecognizer!.decode(_transducerStream!);
          }

          final result = _transducerRecognizer!.getResult(_transducerStream!);
          if (result.text.isNotEmpty) {
            setState(() => _transcribedText = result.text);
          }

          if (_transducerRecognizer!.isEndpoint(_transducerStream!)) {
            _stopListening();
          }
        } else if (sttType == SttModelType.whisper && _whisperStream != null) {
          _whisperStream!.acceptWaveform(samples: float32s, sampleRate: 16000);
        }
      });
    } catch (e) {
      _log("Mic Err: $e");
    }
  }

  Future<void> _stopListening() async {
    final sttType = _activeModelType ?? widget.modelType;

    await _audioSub?.cancel();
    await _recorder.stop();
    setState(() {
      _isListening = false;
      _isProcessing = true;
      _statusMessage = "Processing...";
    });

    if (sttType == SttModelType.whisper && _whisperStream != null) {
      try {
        _whisperRecognizer!.decode(_whisperStream!);
        final result = _whisperRecognizer!.getResult(_whisperStream!);
        _transcribedText = result.text;
      } catch (e) {
        _log("Whisper Decode Error: $e");
      }
    } else if (sttType == SttModelType.transducer &&
        _transducerStream != null) {
      try {
        _transducerStream!.inputFinished();

        while (_transducerRecognizer!.isReady(_transducerStream!)) {
          _transducerRecognizer!.decode(_transducerStream!);
        }
        final result = _transducerRecognizer!.getResult(_transducerStream!);
        if (result.text.isNotEmpty) _transcribedText = result.text;
      } catch (e) {
        _log("Transducer Decode Error: $e");
      }
    }

    if (_transcribedText.isNotEmpty) {
      await _handleCommand(_transcribedText);
    } else {
      setState(() {
        _isProcessing = false;
        _statusMessage = "No speech detected";
      });
    }
  }

  Future<void> _handleCommand(String cmd) async {
    String response = "मुझे समझ नहीं आया";
    cmd = cmd.toLowerCase();

    if (cmd.contains("नमस्ते") || cmd.contains("hello") || cmd.contains("hi")) {
      response = "नमस्ते! मैं जार्विस हूँ।";
    } else if (cmd.contains("time") || cmd.contains("समय")) {
      response = "अभी ${DateTime.now().hour} बजे हैं";
    } else if (cmd.contains("kaise") || cmd.contains("how are you")) {
      response = "मैं ठीक हूँ, धन्यवाद!";
    }

    if (response == "मुझे समझ नहीं आया") response = "आपने कहा: $cmd";

    _log("🤖: $response");
    await _speak(response);
    if (mounted) setState(() => _isProcessing = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
            gradient: LinearGradient(
                colors: [Color(0xFF0A0E27), Color(0xFF16213E)],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter)),
        child: SafeArea(
          child: Column(
            children: [
              const Padding(
                  padding: EdgeInsets.all(20),
                  child: Text("JARVIS",
                      style: TextStyle(
                          fontSize: 30,
                          color: Color(0xFF00E5FF),
                          letterSpacing: 5))),
              Expanded(
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Text(
                      _transcribedText.isEmpty
                          ? "Tap Mic..."
                          : _transcribedText,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                          fontSize: 24,
                          color: _transcribedText.isEmpty
                              ? Colors.white30
                              : Colors.white),
                    ),
                  ),
                ),
              ),
              if (_isProcessing)
                const LinearProgressIndicator(color: Color(0xFF00E5FF)),
              const SizedBox(height: 20),
              GestureDetector(
                onTap: _toggleListening,
                child: AnimatedBuilder(
                  animation: _pulseAnimation,
                  builder: (ctx, child) => Transform.scale(
                    scale: _isListening ? _pulseAnimation.value : 1.0,
                    child: Container(
                      height: 80,
                      width: 80,
                      decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _isListening
                              ? Colors.redAccent
                              : const Color(0xFF00E5FF),
                          boxShadow: [
                            BoxShadow(
                                color: (_isListening ? Colors.red : Colors.blue)
                                    .withOpacity(0.5),
                                blurRadius: 20)
                          ]),
                      child: Icon(_isListening ? Icons.stop : Icons.mic,
                          color: Colors.white, size: 40),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 40),
              Text(_statusMessage,
                  style: const TextStyle(color: Colors.white54)),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
