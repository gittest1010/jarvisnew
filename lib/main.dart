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

/* ==================== INIT SCREEN ==================== */
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
      _log("🔐 Requesting Permissions...");
      await Permission.microphone.request();
      await Permission.storage.request();

      _log("⚙️ Initializing Sherpa ONNX Bindings...");
      sherpa.initBindings();

      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;

      // Extract Assets
      await _extractIfNeeded("assets/stt-hi.tar.bz2", "stt_root", basePath);
      await _extractIfNeeded("assets/tts-hi.tar.bz2", "tts_root", basePath);

      // Find Model Files
      _log("🔍 Locating AI Models...");
      final sttDir = Directory("$basePath/stt_root");
      final ttsDir = Directory("$basePath/tts_root");

      // STT Files (Online Streaming Model)
      final encoder =
          await _recursiveFind(sttDir, "encoder-epoch-99-avg-1.onnx") ??
              await _recursiveFind(sttDir, "encoder.onnx") ??
              await _recursiveFind(sttDir, "encoder.int8.onnx");

      final decoder =
          await _recursiveFind(sttDir, "decoder-epoch-99-avg-1.onnx") ??
              await _recursiveFind(sttDir, "decoder.onnx") ??
              await _recursiveFind(sttDir, "decoder.int8.onnx");

      final joiner =
          await _recursiveFind(sttDir, "joiner-epoch-99-avg-1.onnx") ??
              await _recursiveFind(sttDir, "joiner.onnx") ??
              await _recursiveFind(sttDir, "joiner.int8.onnx");

      final sttTokens = await _recursiveFind(sttDir, "tokens.txt");

      // TTS Files
      final ttsModel = await _recursiveFind(ttsDir, "model.onnx");
      final ttsTokens = await _recursiveFind(ttsDir, "tokens.txt");
      final espeakData =
          await _recursiveFind(ttsDir, "espeak-ng-data", isFolder: true);

      // Validation
      if (encoder == null) throw "❌ STT Encoder not found";
      if (decoder == null) throw "❌ STT Decoder not found";
      if (joiner == null) throw "❌ STT Joiner not found";
      if (sttTokens == null) throw "❌ STT Tokens not found";
      if (ttsModel == null) throw "❌ TTS Model not found";
      if (ttsTokens == null) throw "❌ TTS Tokens not found";
      if (espeakData == null) throw "❌ eSpeak Data not found";

      validPaths = {
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "sttTokens": sttTokens,
        "ttsModel": ttsModel,
        "ttsTokens": ttsTokens,
        "espeakData": espeakData,
      };

      _log("✅ All Systems Ready. Activating Jarvis...");
      await Future.delayed(const Duration(seconds: 1));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => JarvisHome(paths: validPaths)),
        );
      }
    } catch (e, stack) {
      _log("💥 FATAL ERROR: $e", error: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  Future<String?> _recursiveFind(Directory dir, String filename,
      {bool isFolder = false}) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity.path.endsWith(filename)) {
          if (isFolder && entity is Directory) return entity.path;
          if (!isFolder && entity is File && entity.lengthSync() > 0) {
            return entity.path;
          }
        }
      }
    } catch (_) {}
    return null;
  }

  Future<void> _extractIfNeeded(
      String asset, String folderName, String basePath) async {
    final target = Directory("$basePath/$folderName");
    if (await target.exists() && target.listSync().isNotEmpty) {
      _log("✓ $folderName already extracted");
      return;
    }

    _log("📦 Extracting $folderName...");
    final data = await rootBundle.load(asset);
    final bytes = data.buffer.asUint8List();
    await compute(_backgroundUnzip, _UnzipArgs(bytes, basePath, folderName));
    _log("✓ $folderName extracted");
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF0A0E27), Color(0xFF1A1A2E)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!isError)
                const CircularProgressIndicator(
                  color: Color(0xFF00E5FF),
                  strokeWidth: 3,
                ),
              const SizedBox(height: 30),
              Text(
                status,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: isError ? Colors.redAccent : Color(0xFF00E5FF),
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
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

/* ==================== JARVIS INTERFACE ==================== */
class JarvisHome extends StatefulWidget {
  final Map<String, String> paths;
  const JarvisHome({super.key, required this.paths});

  @override
  State<JarvisHome> createState() => _JarvisHomeState();
}

class _JarvisHomeState extends State<JarvisHome>
    with SingleTickerProviderStateMixin {
  // AI Engines
  sherpa.OfflineTts? _tts;
  sherpa.OnlineRecognizer? _recognizer;
  sherpa.OnlineStream? _stream;

  // Audio
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription<Uint8List>? _audioSub;

  // State
  bool _isListening = false;
  bool _isSpeaking = false;
  bool _isProcessing = false;
  String _transcribedText = "";
  String _statusMessage = "Ready";
  String _lastRecognizedText = "";

  // Animation
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
      _log("🧠 Initializing TTS Engine...");
      _tts = sherpa.OfflineTts(
        sherpa.OfflineTtsConfig(
          model: sherpa.OfflineTtsModelConfig(
            vits: sherpa.OfflineTtsVitsModelConfig(
              model: widget.paths["ttsModel"]!,
              tokens: widget.paths["ttsTokens"]!,
              dataDir: widget.paths["espeakData"]!,
            ),
            numThreads: 2,
            debug: false,
            provider: "cpu",
          ),
        ),
      );

      _log("🎙️ Initializing STT Engine...");
      _recognizer = sherpa.OnlineRecognizer(
        sherpa.OnlineRecognizerConfig(
          model: sherpa.OnlineModelConfig(
            transducer: sherpa.OnlineTransducerModelConfig(
              encoder: widget.paths["encoder"]!,
              decoder: widget.paths["decoder"]!,
              joiner: widget.paths["joiner"]!,
            ),
            tokens: widget.paths["sttTokens"]!,
            numThreads: 2,
            debug: false,
            provider: "cpu",
          ),
          enableEndpoint: true,
          rule1MinTrailingSilence: 2.4,
          rule2MinTrailingSilence: 1.2,
          rule3MinUtteranceLength: 20,
        ),
      );

      _log("✅ Jarvis is now online");
      _speak("नमस्ते, मैं जार्विस हूं, आपकी सेवा में हाजिर");
    } catch (e, stack) {
      _log("❌ AI Init Failed: $e");
      debugPrintStack(stackTrace: stack);
    }
  }

  void _log(String msg) {
    debugPrint(msg);
    if (mounted) setState(() => _statusMessage = msg);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _audioSub?.cancel();
    _recorder.dispose();
    _stream = null; // Just set to null, no need to dispose
    _recognizer = null; // Just set to null, no need to dispose
    _tts = null; // Just set to null, no need to dispose
    _audioPlayer.dispose();
    super.dispose();
  }

  // ========== TTS ==========
  Future<void> _speak(String text) async {
    if (text.isEmpty || _tts == null) return;

    setState(() {
      _isSpeaking = true;
      _statusMessage = "🔊 Speaking...";
    });

    try {
      final audio = _tts!.generate(text: text, sid: 0, speed: 1.0);
      final samples = audio.samples;
      final sampleRate = audio.sampleRate;

      // Convert Float32 → Int16 PCM
      final pcm = Int16List(samples.length);
      for (int i = 0; i < samples.length; i++) {
        int val = (samples[i] * 32767).round().clamp(-32768, 32767);
        pcm[i] = val;
      }

      // Create WAV file
      final tempDir = await getTemporaryDirectory();
      final wavPath = "${tempDir.path}/jarvis_output.wav";
      final wavFile = File(wavPath);
      final wavBytes = _createWav(pcm, sampleRate);
      await wavFile.writeAsBytes(wavBytes);

      // Play the audio
      await _audioPlayer.play(DeviceFileSource(wavPath));

      // Wait for playback to complete
      await _audioPlayer.onPlayerComplete.first;

      _log("✅ Speech complete");
    } catch (e) {
      _log("❌ TTS Error: $e");
    } finally {
      if (mounted) {
        setState(() => _isSpeaking = false);
      }
    }
  }

  Uint8List _createWav(Int16List pcm, int sampleRate) {
    final dataSize = pcm.length * 2;
    final byteRate = sampleRate * 2;
    final header = BytesBuilder();

    header.add(Uint8List.fromList("RIFF".codeUnits));
    header.add(_int32Bytes(36 + dataSize));
    header.add(Uint8List.fromList("WAVE".codeUnits));
    header.add(Uint8List.fromList("fmt ".codeUnits));
    header.add(_int32Bytes(16));
    header.add(_int16Bytes(1)); // PCM
    header.add(_int16Bytes(1)); // Mono
    header.add(_int32Bytes(sampleRate));
    header.add(_int32Bytes(byteRate));
    header.add(_int16Bytes(2));
    header.add(_int16Bytes(16));
    header.add(Uint8List.fromList("data".codeUnits));
    header.add(_int32Bytes(dataSize));

    final buffer = BytesBuilder();
    buffer.add(header.toBytes());
    buffer.add(pcm.buffer.asUint8List());

    return buffer.toBytes();
  }

  Uint8List _int32Bytes(int value) {
    return Uint8List(4)..buffer.asByteData().setInt32(0, value, Endian.little);
  }

  Uint8List _int16Bytes(int value) {
    return Uint8List(2)..buffer.asByteData().setInt16(0, value, Endian.little);
  }

  // ========== STT ==========
  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    if (_recognizer == null || _isSpeaking || _isProcessing) return;

    if (!await _recorder.hasPermission()) {
      _log("❌ Microphone permission denied");
      return;
    }

    try {
      _stream = _recognizer!.createStream();

      final config = RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        autoGain: true,
        echoCancel: true,
        noiseSuppress: true,
      );

      final audioStream = await _recorder.startStream(config);

      setState(() {
        _isListening = true;
        _transcribedText = "";
        _lastRecognizedText = "";
        _statusMessage = "🎤 Listening...";
      });

      _audioSub = audioStream.listen((data) {
        _processAudio(data);
      });
    } catch (e) {
      _log("❌ Mic error: $e");
      setState(() => _isListening = false);
    }
  }

  void _processAudio(Uint8List data) {
    if (_stream == null) return;

    // Convert PCM16 → Float32
    final int16 = Int16List.view(data.buffer);
    final float32 = Float32List(int16.length);
    for (int i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    _stream!.acceptWaveform(samples: float32, sampleRate: 16000);

    while (_recognizer!.isReady(_stream!)) {
      _recognizer!.decode(_stream!);
    }

    final result = _recognizer!.getResult(_stream!);
    final text = result.text.trim();

    if (text.isNotEmpty && text != _lastRecognizedText) {
      setState(() {
        _lastRecognizedText = text;
        _transcribedText = text;
      });
    }

    // Auto-stop on endpoint
    if (_recognizer!.isEndpoint(_stream!)) {
      _stopListening();
    }
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    await _recorder.stop();

    setState(() {
      _isListening = false;
      _statusMessage =
          _transcribedText.isEmpty ? "No speech detected" : "Processing...";
      _isProcessing = true;
    });

    if (_transcribedText.isNotEmpty) {
      await _handleCommand(_transcribedText);
    } else {
      setState(() => _isProcessing = false);
    }
  }

  // ========== BRAIN ==========
  Future<void> _handleCommand(String cmd) async {
    String response = "";
    cmd = cmd.toLowerCase();

    if (cmd.contains("नमस्ते") || cmd.contains("hello")) {
      response = "नमस्ते, मैं आपकी क्या मदद कर सकता हूं?";
    } else if (cmd.contains("समय") || cmd.contains("time")) {
      final now = DateTime.now();
      response = "अभी ${now.hour} बजकर ${now.minute} मिनट हुए हैं";
    } else if (cmd.contains("तारीख") || cmd.contains("date")) {
      final now = DateTime.now();
      response = "आज ${now.day} ${_getMonthName(now.month)} ${now.year} है";
    } else if (cmd.contains("धन्यवाद") || cmd.contains("thank")) {
      response = "आपका स्वागत है";
    } else {
      response =
          "आपने कहा: $cmd। मैं अभी सीख रहा हूँ, इसलिए कुछ कमांड ही समझ पाता हूँ।";
    }

    _log("🤖 $response");
    await _speak(response);
    if (mounted) {
      setState(() => _isProcessing = false);
    }
  }

  String _getMonthName(int month) {
    const months = [
      "जनवरी",
      "फरवरी",
      "मार्च",
      "अप्रैल",
      "मई",
      "जून",
      "जुलाई",
      "अगस्त",
      "सितंबर",
      "अक्टूबर",
      "नवंबर",
      "दिसंबर"
    ];
    return months[month - 1];
  }

  // ========== UI ==========
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF0A0E27), Color(0xFF16213E)],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.all(20),
                child: Text(
                  "J A R V I S",
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w300,
                    letterSpacing: 8,
                    color: Color(0xFF00E5FF),
                  ),
                ),
              ),

              // Transcript Display
              Expanded(
                child: Container(
                  margin: EdgeInsets.symmetric(horizontal: 20),
                  padding: EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.03),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: SingleChildScrollView(
                    reverse: true,
                    child: Column(
                      children: [
                        Text(
                          _transcribedText.isEmpty
                              ? "Tap mic to speak"
                              : _transcribedText,
                          style: TextStyle(
                            fontSize: 20,
                            color: _transcribedText.isEmpty
                                ? Colors.white38
                                : Color(0xFF00E5FF),
                            height: 1.5,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        if (_isProcessing)
                          Padding(
                            padding: const EdgeInsets.only(top: 20),
                            child: CircularProgressIndicator(
                              color: Color(0xFF00E5FF),
                              strokeWidth: 2,
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ),

              // Status
              Padding(
                padding: EdgeInsets.all(20),
                child: Text(
                  _statusMessage,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white60,
                  ),
                ),
              ),

              // Mic Button
              Padding(
                padding: EdgeInsets.only(bottom: 50),
                child: GestureDetector(
                  onTap: _isSpeaking || _isProcessing ? null : _toggleListening,
                  child: AnimatedBuilder(
                    animation: _pulseAnimation,
                    builder: (context, child) {
                      return Transform.scale(
                        scale: _isListening ? _pulseAnimation.value : 1.0,
                        child: Container(
                          height: 90,
                          width: 90,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            gradient: LinearGradient(
                              colors: _isListening
                                  ? [Color(0xFFFF1744), Color(0xFFD50000)]
                                  : [Color(0xFF00E5FF), Color(0xFF0091EA)],
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: (_isListening
                                        ? Color(0xFFFF1744)
                                        : Color(0xFF00E5FF))
                                    .withOpacity(0.6),
                                blurRadius: 30,
                                spreadRadius: _isListening ? 10 : 5,
                              ),
                            ],
                          ),
                          child: Icon(
                            _isListening
                                ? Icons.stop_rounded
                                : Icons.mic_rounded,
                            size: 45,
                            color: Colors.white,
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
