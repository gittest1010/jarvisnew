import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
// Fixed: Removed duplicate import 'package:archive/archive.dart'
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

/* ==================== INIT SCREEN (SETUP LOGIC) ==================== */
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
      _log("🔐 Permissions check...");
      await Permission.microphone.request();
      await Permission.storage.request(); // Important for some Android versions

      _log("⚙️ Init Sherpa Bindings...");
      sherpa.initBindings();

      final docDir = await getApplicationDocumentsDirectory();
      final basePath = docDir.path;

      // Define paths
      final sttDir = Directory("$basePath/stt_root");
      final ttsDir = Directory("$basePath/tts_root");

      // --- PHASE 1: EXTRACTION ---
      await _extractIfNeeded("assets/stt-hi.tar.bz2", "stt_root", basePath);
      await _extractIfNeeded("assets/tts-hi.tar.bz2", "tts_root", basePath);

      // --- PHASE 2: FINDING FILES (With Auto-Repair) ---
      _log("🔍 Locating AI Models...");

      // Try finding files. If critical files are missing, we force re-extract ONCE.
      bool filesMissing = false;

      // Check STT basic existence
      if (await _findAny(sttDir, ["encoder", "encoder.onnx"]) == null)
        filesMissing = true;
      if (await _findAny(sttDir, ["tokens.txt"]) == null) filesMissing = true;

      // Check TTS basic existence
      if (await _findAny(ttsDir, ["model.onnx"]) == null) filesMissing = true;

      if (filesMissing) {
        _log("⚠️ Corrupted or missing files detected. Re-installing...");
        if (sttDir.existsSync()) sttDir.deleteSync(recursive: true);
        if (ttsDir.existsSync()) ttsDir.deleteSync(recursive: true);

        // Re-extract fresh
        await _extractIfNeeded("assets/stt-hi.tar.bz2", "stt_root", basePath,
            force: true);
        await _extractIfNeeded("assets/tts-hi.tar.bz2", "tts_root", basePath,
            force: true);
      }

      // --- PHASE 3: FINAL ASSIGNMENT ---
      // STT (Streaming Zipformer)
      final encoder = await _findAny(sttDir,
          ["encoder-epoch-99-avg-1.onnx", "encoder.onnx", "encoder.int8.onnx"]);

      final decoder = await _findAny(sttDir,
          ["decoder-epoch-99-avg-1.onnx", "decoder.onnx", "decoder.int8.onnx"]);

      final joiner = await _findAny(sttDir,
          ["joiner-epoch-99-avg-1.onnx", "joiner.onnx", "joiner.int8.onnx"]);

      final sttTokens = await _findAny(sttDir, ["tokens.txt"]);

      // TTS (VITS)
      final ttsModel =
          await _findAny(ttsDir, ["model.onnx", "vits-model.onnx"]);
      final ttsTokens = await _findAny(ttsDir, ["tokens.txt"]);
      final espeakData = await _findFolder(ttsDir, "espeak-ng-data");

      // --- PHASE 4: VALIDATION ---
      if (encoder == null) throw _errorMsg(sttDir, "STT Encoder");
      if (decoder == null) throw _errorMsg(sttDir, "STT Decoder");
      if (joiner == null) throw _errorMsg(sttDir, "STT Joiner");
      if (sttTokens == null) throw _errorMsg(sttDir, "STT Tokens (tokens.txt)");

      if (ttsModel == null) throw _errorMsg(ttsDir, "TTS Model");
      if (ttsTokens == null) throw _errorMsg(ttsDir, "TTS Tokens (tokens.txt)");
      if (espeakData == null) throw _errorMsg(ttsDir, "eSpeak Data Folder");

      // Verify no cross-talk (Conflict Check)
      if (!sttTokens.contains("stt_root"))
        _log("⚠️ Warning: STT tokens path looks wrong: $sttTokens");
      if (!ttsTokens.contains("tts_root"))
        _log("⚠️ Warning: TTS tokens path looks wrong: $ttsTokens");

      _log(
          "✅ Found STT Tokens: ...${sttTokens.substring(sttTokens.length > 20 ? sttTokens.length - 20 : 0)}");
      _log(
          "✅ Found TTS Tokens: ...${ttsTokens.substring(ttsTokens.length > 20 ? ttsTokens.length - 20 : 0)}");

      validPaths = {
        "encoder": encoder,
        "decoder": decoder,
        "joiner": joiner,
        "sttTokens": sttTokens,
        "ttsModel": ttsModel,
        "ttsTokens": ttsTokens,
        "espeakData": espeakData,
      };

      _log("🚀 Starting Jarvis...");
      await Future.delayed(const Duration(seconds: 1));

      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => JarvisHome(paths: validPaths)),
        );
      }
    } catch (e, stack) {
      _log("💥 ERROR: $e", error: true);
      debugPrintStack(stackTrace: stack);
    }
  }

  String _errorMsg(Directory dir, String missing) {
    // Dump files to help user debug
    _listAllFiles(dir);
    return "Could not find $missing in ${dir.path.split('/').last}";
  }

  void _listAllFiles(Directory dir) {
    if (!dir.existsSync()) {
      _log("❌ Directory missing: ${dir.path}");
      return;
    }
    _log("📂 Files in ${dir.path.split('/').last}:");
    try {
      final list = dir.listSync(recursive: true);
      for (var f in list) {
        if (f is File) _log(" - ${f.path.split('/').last}");
      }
    } catch (e) {
      _log("Error listing files: $e");
    }
  }

  Future<String?> _findAny(Directory dir, List<String> possibilities) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity is File) {
          final name = entity.path.split('/').last.toLowerCase();
          for (var p in possibilities) {
            // Check if name ends with possibility (handles minor naming diffs)
            if (name == p.toLowerCase() || name.endsWith(p.toLowerCase())) {
              return entity.path;
            }
          }
        }
      }
    } catch (_) {}
    return null;
  }

  Future<String?> _findFolder(Directory dir, String folderName) async {
    if (!await dir.exists()) return null;
    try {
      final entities = dir.listSync(recursive: true);
      for (var entity in entities) {
        if (entity is Directory) {
          if (entity.path.split('/').last == folderName) {
            return entity.path;
          }
        }
      }
    } catch (_) {}
    return null;
  }

  Future<void> _extractIfNeeded(
      String asset, String folderName, String basePath,
      {bool force = false}) async {
    final target = Directory("$basePath/$folderName");

    if (!force && await target.exists() && target.listSync().isNotEmpty) {
      _log("✓ $folderName ready");
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
        padding: EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF0A0E27), Color(0xFF1A1A2E)],
          ),
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
                color: isError ? Colors.redAccent : Color(0xFF00E5FF),
                fontSize: 16,
              ),
            ),
            if (isError) ...[
              SizedBox(height: 20),
              Expanded(
                child: Container(
                  padding: EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    border: Border.all(color: Colors.redAccent),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SingleChildScrollView(
                    child: Text(logs,
                        style:
                            TextStyle(fontFamily: 'monospace', fontSize: 11)),
                  ),
                ),
              ),
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

/* ==================== JARVIS INTERFACE ==================== */
class JarvisHome extends StatefulWidget {
  final Map<String, String> paths;
  const JarvisHome({super.key, required this.paths});

  @override
  State<JarvisHome> createState() => _JarvisHomeState();
}

class _JarvisHomeState extends State<JarvisHome>
    with SingleTickerProviderStateMixin {
  sherpa.OfflineTts? _tts;
  sherpa.OnlineRecognizer? _recognizer;
  sherpa.OnlineStream? _stream;

  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  StreamSubscription<Uint8List>? _audioSub;

  bool _isListening = false;
  bool _isSpeaking = false;
  bool _isProcessing = false;
  String _transcribedText = "";
  String _statusMessage = "Ready";
  String _lastRecognizedText = "";

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
      _log("🧠 Starting Engines...");

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
          ),
        ),
      );

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
          ),
          enableEndpoint: true,
          rule1MinTrailingSilence: 2.4,
          rule2MinTrailingSilence: 1.2,
          rule3MinUtteranceLength: 20,
        ),
      );

      _log("✅ Online");
      _speak("नमस्ते, मैं तैयार हूं");
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
    _stream = null;
    _recognizer = null;
    _tts = null;
    _audioPlayer.dispose();
    super.dispose();
  }

  // ... TTS Logic ...
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
      final wavFile = File(wavPath);
      await wavFile.writeAsBytes(_createWav(pcm, audio.sampleRate));

      await _audioPlayer.play(DeviceFileSource(wavPath));
      await _audioPlayer.onPlayerComplete.first;
    } catch (e) {
      _log("TTS Err: $e");
    } finally {
      if (mounted) setState(() => _isSpeaking = false);
    }
  }

  // ... WAV Header Helper ...
  Uint8List _createWav(Int16List pcm, int sampleRate) {
    var channels = 1;
    var byteRate = sampleRate * channels * 2;
    var header = BytesBuilder();
    header.add(Uint8List.fromList("RIFF".codeUnits));
    header.add(_int32Bytes(36 + pcm.length * 2));
    header.add(Uint8List.fromList("WAVEfmt ".codeUnits));
    header.add(_int32Bytes(16));
    header.add(_int16Bytes(1));
    header.add(_int16Bytes(channels));
    header.add(_int32Bytes(sampleRate));
    header.add(_int32Bytes(byteRate));
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

  // ... STT Logic ...
  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    if (_recognizer == null || _isSpeaking) return;
    if (!await _recorder.hasPermission()) return;

    try {
      _stream = _recognizer!.createStream();

      // Capture the stream returned by startStream
      // Using RecordConfig from record package v6.1.2
      final stream = await _recorder.startStream(RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        echoCancel: true,
        noiseSuppress: true,
      ));

      setState(() {
        _isListening = true;
        _transcribedText = "";
        _statusMessage = "🎤 Listening...";
      });

      // Listen to the captured stream instead of onStream
      _audioSub = stream.listen((data) {
        // Convert Uint8List (PCM16) -> Float32List
        final int16s = Int16List.view(data.buffer);
        final float32s = Float32List(int16s.length);
        for (int i = 0; i < int16s.length; i++)
          float32s[i] = int16s[i] / 32768.0;

        _stream!.acceptWaveform(samples: float32s, sampleRate: 16000);
        while (_recognizer!.isReady(_stream!)) {
          _recognizer!.decode(_stream!);
        }
        final result = _recognizer!.getResult(_stream!);
        if (result.text.isNotEmpty) {
          setState(() => _transcribedText = result.text);
        }
        if (_recognizer!.isEndpoint(_stream!)) _stopListening();
      });
    } catch (e) {
      _log("Mic Err: $e");
      setState(() => _isListening = false);
    }
  }

  Future<void> _stopListening() async {
    await _audioSub?.cancel();
    await _recorder.stop();
    setState(() {
      _isListening = false;
      _isProcessing = true;
    });

    if (_transcribedText.isNotEmpty) {
      await _handleCommand(_transcribedText);
    } else {
      setState(() => _isProcessing = false);
    }
  }

  Future<void> _handleCommand(String cmd) async {
    String response = "मुझे समझ नहीं आया";
    cmd = cmd.toLowerCase();
    if (cmd.contains("नमस्ते")) response = "नमस्ते! कहिये क्या सेवा करूँ?";
    if (cmd.contains("समय")) response = "अभी ${DateTime.now().hour} बजे हैं";

    _log("🤖: $response");
    await _speak(response);
    if (mounted) setState(() => _isProcessing = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
            gradient: LinearGradient(
                colors: [Color(0xFF0A0E27), Color(0xFF16213E)],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter)),
        child: SafeArea(
          child: Column(
            children: [
              Padding(
                  padding: EdgeInsets.all(20),
                  child: Text("JARVIS",
                      style: TextStyle(
                          fontSize: 30,
                          color: Color(0xFF00E5FF),
                          letterSpacing: 5))),
              Expanded(
                child: Center(
                  child: Padding(
                    padding: EdgeInsets.all(20),
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
                LinearProgressIndicator(color: Color(0xFF00E5FF)),
              SizedBox(height: 20),
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
                              : Color(0xFF00E5FF),
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
              SizedBox(height: 40),
              Text(_statusMessage, style: TextStyle(color: Colors.white54)),
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
