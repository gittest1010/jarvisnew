import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';
import 'package:archive/archive_io.dart';
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

// Model Type define karne ke liye enum
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
  SttModelType detectedModelType = SttModelType.transducer; // Default

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
      sherpa.initBindings();

      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;
      final sttDir = Directory("$basePath/stt_root");
      final ttsDir = Directory("$basePath/tts_root");

      // --- PHASE 1: EXTRACTION ---
      await _extractIfNeeded("assets/stt-hi.tar.bz2", "stt_root", basePath);
      await _extractIfNeeded("assets/tts-hi.tar.bz2", "tts_root", basePath);

      // --- PHASE 2: SMART DETECTION (The Fix) ---
      _log("🔍 Detecting Model Architecture...");

      // Pehle Encoder dhundte hain (Dono models me hota hai)
      final encoder = await _smartFind(sttDir, [
        "encoder.onnx",
        "encoder.int8.onnx",
        "tiny-encoder.onnx",
        "base-encoder.onnx"
      ]);

      // Fir Joiner dhundte hain
      final joiner = await _smartFind(sttDir,
          ["joiner.onnx", "joiner.int8.onnx", "joiner-epoch-99-avg-1.onnx"]);

      // Decoder dhundte hain
      final decoder = await _smartFind(sttDir, [
        "decoder.onnx",
        "decoder.int8.onnx",
        "tiny-decoder.onnx",
        "base-decoder.onnx"
      ]);

      // Tokens dhundte hain
      final sttTokens = await _smartFind(sttDir, ["tokens.txt"]);

      // --- LOGIC: JOINER NAHI TO WHISPER ---
      if (encoder != null && joiner == null) {
        _log("⚠️ Joiner not found. Assuming WHISPER Model.");
        detectedModelType = SttModelType.whisper;
      } else if (encoder != null && joiner != null) {
        _log("✅ Joiner found. Assuming ZIPFORMER/TRANSDUCER Model.");
        detectedModelType = SttModelType.transducer;
      } else {
        throw "❌ No valid STT model found in stt_root. (Encoder missing)";
      }

      // --- TTS Files ---
      final ttsModel =
          await _smartFind(ttsDir, ["model.onnx", "vits-model.onnx"]);
      final ttsTokens = await _smartFind(ttsDir, ["tokens.txt"]);
      final espeakData = await _smartFindFolder(ttsDir, "espeak-ng-data");
      // Whisper me joiner zaruri nahi hai, isliye check skip kar rahe hain agar whisper mode hai
      if (detectedModelType == SttModelType.transducer && joiner == null)
        throw "STT Joiner Missing (Required for Zipformer)!";
      if (decoder == null) throw "STT Decoder Missing!";
      if (sttTokens == null) throw "STT Tokens Missing!";

      if (ttsModel == null) throw "TTS Model Missing!";
      if (ttsTokens == null) throw "TTS Tokens Missing!";
      if (espeakData == null) throw "eSpeak Data Missing!";

      validPaths = {
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner ?? "", // Empty string if whisper
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
              modelType: detectedModelType, // Pass model type
            ),
          ),
        );
      }
    } catch (e, stack) {
      _log("💥 ERROR: $e", error: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  // Helpers (Same as before)
  Future<String?> _smartFind(Directory dir, List<String> patterns) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity is File) {
          final name = entity.path.split('/').last.toLowerCase();
          for (var p in patterns) {
            if (name.contains(p.toLowerCase())) return entity.path;
          }
        }
      }
    } catch (_) {}
    return null;
  }

  Future<String?> _smartFindFolder(Directory dir, String folderName) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity is Directory) {
          if (entity.path.split('/').last == folderName) return entity.path;
        }
      }
    } catch (_) {}
    return null;
  }

  Future<void> _extractIfNeeded(
      String asset, String folderName, String basePath) async {
    final target = Directory("$basePath/$folderName");
    if (await target.exists() && target.listSync().isNotEmpty) {
      _log("✓ $folderName ready");
      return;
    }
    _log("📦 Extracting $folderName...");
    final data = await rootBundle.load(asset);
    final bytes = data.buffer.asUint8List();
    await compute(_backgroundUnzip, _UnzipArgs(bytes, basePath, folderName));
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
            Text(status,
                textAlign: TextAlign.center,
                style: TextStyle(
                    color:
                        isError ? Colors.redAccent : const Color(0xFF00E5FF))),
            if (isError) ...[
              const SizedBox(height: 20),
              Expanded(
                child: Container(
                  padding: const EdgeInsets.all(10),
                  color: Colors.black54,
                  child: SingleChildScrollView(
                      child: Text(logs,
                          style: const TextStyle(
                              fontFamily: 'monospace', fontSize: 11))),
                ),
              )
            ]
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

  // STT Engines (Only one will be active)
  sherpa.OnlineRecognizer? _transducerRecognizer;
  sherpa.OnlineStream? _transducerStream;

  sherpa.OfflineRecognizer? _whisperRecognizer;
  sherpa.OfflineStream? _whisperStream; // Whisper stream is different

  final AudioRecorder _recorder = AudioRecorder();
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

      // 1. Setup TTS (Same for all)
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

      // 2. Setup STT based on Type
      if (widget.modelType == SttModelType.transducer) {
        // Zipformer / Transducer (Requires Joiner)
        _transducerRecognizer = sherpa.OnlineRecognizer(
          sherpa.OnlineRecognizerConfig(
            model: sherpa.OnlineModelConfig(
              transducer: sherpa.OnlineTransducerModelConfig(
                encoder: widget.paths["encoder"]!,
                decoder: widget.paths["decoder"]!,
                joiner: widget.paths["joiner"]!,
              ),
              tokens: widget.paths["sttTokens"]!,
              numThreads: 1,
            ),
            enableEndpoint: true,
          ),
        );
      } else {
        // Whisper (No Joiner)
        _whisperRecognizer = sherpa.OfflineRecognizer(
          sherpa.OfflineRecognizerConfig(
            model: sherpa.OfflineModelConfig(
              whisper: sherpa.OfflineWhisperModelConfig(
                encoder: widget.paths["encoder"]!,
                decoder: widget.paths["decoder"]!,
              ),
              tokens: widget.paths["sttTokens"]!,
              numThreads: 1,
              debug: false,
            ),
          ),
        );
      }

      _log("✅ Online");
      _speak("System ready. Using ${widget.modelType.name} model.");
    } catch (e) {
      _log("❌ Init Error: $e");
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
    _transducerStream?.free(); // Important for C++ cleanup
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
      final audio = _tts!.generate(text: text, sid: 0, speed: 1.0);
      final pcm = Int16List(audio.samples.length);
      for (int i = 0; i < audio.samples.length; i++) {
        pcm[i] = (audio.samples[i] * 32767).round().clamp(-32768, 32767);
      }
      final tempDir = await getTemporaryDirectory();
      final wavPath = "${tempDir.path}/jarvis_out.wav";
      File(wavPath).writeAsBytesSync(_createWav(pcm, audio.sampleRate));
      await _audioPlayer.play(DeviceFileSource(wavPath));
      await _audioPlayer.onPlayerComplete.first;
    } catch (_) {}
    if (mounted) setState(() => _isSpeaking = false);
  }

  Uint8List _createWav(Int16List pcm, int sampleRate) {
    var header = BytesBuilder();
    header.add(Uint8List.fromList("RIFF".codeUnits));
    header.add(_int32Bytes(36 + pcm.length * 2));
    header.add(Uint8List.fromList("WAVEfmt ".codeUnits));
    header.add(_int32Bytes(16));
    header.add(_int16Bytes(1));
    header.add(_int16Bytes(1));
    header.add(_int32Bytes(sampleRate));
    header.add(_int32Bytes(sampleRate * 2));
    header.add(_int16Bytes(2));
    header.add(_int16Bytes(16));
    header.add(Uint8List.fromList("data".codeUnits));
    header.add(_int32Bytes(pcm.length * 2));
    var buffer = BytesBuilder();
    buffer.add(header.toBytes());
    buffer.add(pcm.buffer.asUint8List());
    return buffer.toBytes();
  }

  Uint8List _int32Bytes(int v) =>
      Uint8List(4)..buffer.asByteData().setInt32(0, v, Endian.little);
  Uint8List _int16Bytes(int v) =>
      Uint8List(2)..buffer.asByteData().setInt16(0, v, Endian.little);

  // ========== DUAL ENGINE LISTENING LOGIC ==========
  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    if ((_transducerRecognizer == null && _whisperRecognizer == null) ||
        _isSpeaking) return;
    if (!await _recorder.hasPermission()) return;

    // Create Stream based on Model Type
    if (widget.modelType == SttModelType.transducer) {
      _transducerStream = _transducerRecognizer!.createStream();
    } else {
      _whisperStream = _whisperRecognizer!.createStream();
    }

    try {
      await _recorder.startStream(const RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        echoCancel: true,
        noiseSuppress: true,
      ));

      setState(() {
        _isListening = true;
        _transcribedText = "";
        _statusMessage = "🎤 Listening (${widget.modelType.name})...";
      });

      _audioSub = _recorder.onStream!.listen((data) {
        // Common PCM -> Float32 Conversion
        final int16s = Int16List.view(data.buffer);
        final float32s = Float32List(int16s.length);
        for (int i = 0; i < int16s.length; i++) {
          float32s[i] = int16s[i] / 32768.0;
        }

        if (widget.modelType == SttModelType.transducer) {
          // --- Transducer (Realtime Streaming) ---
          _transducerStream!
              .acceptWaveform(samples: float32s, sampleRate: 16000);
          while (_transducerRecognizer!.isReady(_transducerStream!)) {
            _transducerRecognizer!.decode(_transducerStream!);
          }
          final result = _transducerRecognizer!.getResult(_transducerStream!);
          if (result.text.isNotEmpty)
            setState(() => _transcribedText = result.text);
          if (_transducerRecognizer!.isEndpoint(_transducerStream!))
            _stopListening();
        } else {
          // --- Whisper (Buffered/Offline) ---
          // Whisper accepts waveform but usually decodes later.
          // However, we can feed it chunks.
          _whisperStream!.acceptWaveform(samples: float32s, sampleRate: 16000);
          // Whisper usually doesn't update partials well in real-time loop without high CPU usage
        }
      });
    } catch (e) {
      _log("Mic Err: $e");
    }
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    await _recorder.stop();
    setState(() {
      _isListening = false;
      _isProcessing = true;
      _statusMessage = "Processing...";
    });

    // Final Decode
    if (widget.modelType == SttModelType.whisper && _whisperStream != null) {
      // For Whisper, we decode once at the end
      _whisperRecognizer!.decode(_whisperStream!);
      final result = _whisperRecognizer!.getResult(_whisperStream!);
      _transcribedText = result.text;
    } else if (widget.modelType == SttModelType.transducer &&
        _transducerStream != null) {
      // Final result already captured, but ensure cleanup
      final result = _transducerRecognizer!.getResult(_transducerStream!);
      if (result.text.isNotEmpty) _transcribedText = result.text;
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

    // Simple commands
    if (cmd.contains("नमस्ते") || cmd.contains("hello")) {
      response = "नमस्ते! मैं जार्विस हूँ।";
    } else if (cmd.contains("time") || cmd.contains("समय"))
      response = "अभी ${DateTime.now().hour} बजे हैं";
    else if (cmd.contains("kaise") || cmd.contains("how are you"))
      response = "मैं ठीक हूँ, धन्यवाद!";

    // Echo if unknown
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
