import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart'; // Correct import for AssetManifest
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:permission_handler/permission_handler.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:path/path.dart' as p;

// Archive package for Zip extraction
import 'package:archive/archive.dart';
import 'package:archive/archive_io.dart';

// ============================================================
// OFFICIAL UTILS CODE (Integrated & Fixed)
// ============================================================

Future<List<String>> getAllAssetFiles() async {
  final AssetManifest assetManifest =
      await AssetManifest.loadFromAssetBundle(rootBundle);
  final List<String> assets = assetManifest.listAssets();
  return assets;
}

String stripLeadingDirectory(String src, {int n = 1}) {
  // Robust check to prevent errors if path is too short
  final parts = p.split(src);
  if (parts.length <= n) {
    return p.basename(src);
  }
  return p.joinAll(parts.sublist(n));
}

Future<void> copyAllAssetFiles() async {
  final allFiles = await getAllAssetFiles();
  for (final src in allFiles) {
    // Determine destination path properly
    // assets/tiny-encoder.onnx -> tiny-encoder.onnx
    // assets/espeak-ng-data/lang/en -> espeak-ng-data/lang/en
    String dst;
    if (src.startsWith('assets/')) {
      dst = src.replaceFirst('assets/', '');
    } else {
      dst = p.basename(src);
    }

    await copyAssetFile(src, dst);
  }
}

Future<String> copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationSupportDirectory();
  if (dst == null) {
    dst = p.basename(src);
  }
  final target = p.join(directory.path, dst);

  // Create parent directory if it doesn't exist
  final File targetFile = File(target);
  if (!await targetFile.parent.exists()) {
    await targetFile.parent.create(recursive: true);
  }

  bool exists = await targetFile.exists();

  // Load asset data
  try {
    final data = await rootBundle.load(src);

    // Copy if file doesn't exist or size is different
    if (!exists || targetFile.lengthSync() != data.lengthInBytes) {
      final List<int> bytes =
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
      await targetFile.writeAsBytes(bytes, flush: true);
      debugPrint("Copied asset: $src -> $target");
    }
  } catch (e) {
    debugPrint("Skipping asset $src (maybe directory or invalid): $e");
  }

  return target;
}

Future<String> generateWaveFilename([String suffix = '']) async {
  final Directory directory = await getApplicationSupportDirectory();
  DateTime now = DateTime.now();
  final filename =
      '${now.year.toString()}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}-${now.hour.toString().padLeft(2, '0')}-${now.minute.toString().padLeft(2, '0')}-${now.second.toString().padLeft(2, '0')}$suffix.wav';

  return p.join(directory.path, filename);
}

// ============================================================
// OFFICIAL MODEL CREATION CODE
// ============================================================

Future<sherpa_onnx.OfflineTts> createOfficialOfflineTts() async {
  // 1. Copy all assets first
  await copyAllAssetFiles();

  // 2. Initialize Bindings
  sherpa_onnx.initBindings();

  final Directory directory = await getApplicationSupportDirectory();

  // Paths
  String modelName = 'hi_IN-pratham-medium.onnx';
  String tokens = 'hi_IN-pratham-medium.onnx.json';
  String dataDir = 'espeak-ng-data';

  // Zip Extraction Logic (Robust)
  // Check both location (assets root and support dir) just in case
  File zipFile = File(p.join(directory.path, 'espeak-ng-data.tar.bz2'));

  final targetDir = Directory(p.join(directory.path, dataDir));

  // Extract only if target folder is missing or empty
  if (!await targetDir.exists() || (await targetDir.list().isEmpty)) {
    if (await zipFile.exists()) {
      debugPrint("Extracting espeak-ng-data.tar.bz2...");
      final bytes = await zipFile.readAsBytes();

      // Compute function use kar rahe hain taaki UI freeze na ho
      final archive = await compute(_decodeArchiveInIsolate, bytes);

      for (final file in archive) {
        final filename = file.name;
        if (file.isFile) {
          final data = file.content as List<int>;
          File(p.join(directory.path, filename))
            ..createSync(recursive: true)
            ..writeAsBytesSync(data);
        }
      }
      debugPrint("Extraction complete.");
    } else {
      debugPrint("Warning: espeak-ng-data.tar.bz2 not found in Support Dir.");
    }
  }

  modelName = p.join(directory.path, modelName);
  tokens = p.join(directory.path, tokens);
  dataDir = p.join(directory.path, dataDir);

  final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
    model: modelName,
    lexicon: '',
    tokens: tokens,
    dataDir: dataDir,
  );

  final kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig();

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    vits: vits,
    kokoro: kokoro,
    numThreads: 2,
    debug: true,
    provider: 'cpu',
  );

  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    ruleFsts: '',
    ruleFars: '',
    maxNumSenetences: 1,
  );

  final tts = sherpa_onnx.OfflineTts(config);
  debugPrint('Offline TTS created successfully');

  return tts;
}

// Top-level function for isolate
Archive _decodeArchiveInIsolate(List<int> bytes) {
  final decoded = BZip2Decoder().decodeBytes(bytes);
  return TarDecoder().decodeBytes(decoded);
}

// ============================================================
// MAIN APP
// ============================================================

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(
      debugShowCheckedModeBanner: false, home: JarvisScreen()));
}

class JarvisScreen extends StatefulWidget {
  const JarvisScreen({super.key});

  @override
  State<JarvisScreen> createState() => _JarvisScreenState();
}

class _JarvisScreenState extends State<JarvisScreen> {
  final TextEditingController _controller = TextEditingController();
  String _statusText = "Initializing...";
  bool _isLoading = true;

  late final AudioRecorder _audioRecorder;
  late final AudioPlayer _audioPlayer;

  sherpa_onnx.OnlineRecognizer? _sttRecognizer;
  sherpa_onnx.OnlineStream? _sttStream;
  sherpa_onnx.OfflineTts? _ttsEngine;

  bool _isRecording = false;
  String _lastRecognizedText = '';
  int _sentenceIndex = 0;
  final int _sampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _audioRecorder = AudioRecorder();
    _audioPlayer = AudioPlayer();
    _startSetup();
  }

  Future<void> _startSetup() async {
    try {
      var status = await Permission.microphone.request();
      if (status != PermissionStatus.granted) {
        setState(() {
          _statusText = "Mic Permission Denied ❌";
          _isLoading = false;
        });
        return;
      }
      await _initJarvis();
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = "Critical Error: $e";
      });
      debugPrint("Startup Error: $e");
    }
  }

  Future<void> _initJarvis() async {
    try {
      setState(() => _statusText = "Preparing Assets (Official)...");

      // 1. Init TTS (This handles copying & initBindings)
      _ttsEngine = await createOfficialOfflineTts();

      setState(() => _statusText = "Initializing STT...");

      // 2. Init STT
      final Directory directory = await getApplicationSupportDirectory();

      final encoderPath = p.join(directory.path, 'tiny-encoder.int8.onnx');
      final decoderPath = p.join(directory.path, 'tiny-decoder.int8.onnx');
      final tokensPath = p.join(directory.path, 'tokens.txt');

      final sttConfig = sherpa_onnx.OnlineRecognizerConfig(
        model: sherpa_onnx.OnlineModelConfig(
          transducer: sherpa_onnx.OnlineTransducerModelConfig(
            encoder: encoderPath,
            decoder: decoderPath,
            joiner: tokensPath,
          ),
          tokens: tokensPath,
          numThreads: 1,
        ),
      );
      _sttRecognizer = sherpa_onnx.OnlineRecognizer(sttConfig);

      setState(() {
        _isLoading = false;
        _statusText = "Jarvis Ready. Tap Mic 🎙️";
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _statusText = "Setup Failed: $e";
      });
      debugPrint("CRITICAL INIT ERROR: $e");
    }
  }

  Future<void> _toggleRecording() async {
    if (_sttRecognizer == null || _isLoading) return;

    try {
      if (_isRecording) {
        await _audioRecorder.stop();
        setState(() => _isRecording = false);
      } else {
        _sttStream?.free();
        _sttStream = _sttRecognizer?.createStream();

        if (await _audioRecorder.hasPermission()) {
          setState(() => _isRecording = true);

          final stream = await _audioRecorder.startStream(const RecordConfig(
            encoder: AudioEncoder.wav,
            sampleRate: 16000,
            numChannels: 1,
          ));

          stream.listen((data) {
            final samples = convertBytesToFloat32(Uint8List.fromList(data));

            if (_sttStream != null) {
              _sttStream!
                  .acceptWaveform(samples: samples, sampleRate: _sampleRate);

              while (_sttRecognizer!.isReady(_sttStream!)) {
                _sttRecognizer!.decode(_sttStream!);
              }

              final text = _sttRecognizer!.getResult(_sttStream!).text;
              bool isEndpoint = _sttRecognizer!.isEndpoint(_sttStream!);
              _updateUI(text, isEndpoint);

              if (isEndpoint) {
                _sttRecognizer!.reset(_sttStream!);
              }
            }
          });
        }
      }
    } catch (e) {
      debugPrint("Recording Error: $e");
    }
  }

  void _updateUI(String text, bool isEndpoint) {
    if (text.isNotEmpty) {
      String fullText = '$_sentenceIndex: $text\n$_lastRecognizedText';

      if (!mounted) return;

      setState(() {
        _controller.value = TextEditingValue(
          text: fullText,
          selection: TextSelection.collapsed(offset: fullText.length),
        );
      });

      if (isEndpoint) {
        _lastRecognizedText = fullText;
        _sentenceIndex++;
        _speak(text);
      }
    }
  }

  Future<void> _speak(String text) async {
    if (_ttsEngine == null) return;
    try {
      final audio = _ttsEngine!.generate(text: text, sid: 0, speed: 1.0);
      if (audio.samples.isEmpty) return;

      final filename = await generateWaveFilename();

      final ok = sherpa_onnx.writeWave(
        filename: filename,
        samples: audio.samples,
        sampleRate: audio.sampleRate,
      );

      if (ok) {
        await _audioPlayer.play(DeviceFileSource(filename));
      }
    } catch (e) {
      debugPrint("Speaking Error: $e");
    }
  }

  Float32List convertBytesToFloat32(Uint8List bytes) {
    final int length = bytes.length ~/ 2;
    final Float32List float32List = Float32List(length);
    final ByteData byteData = ByteData.sublistView(bytes);

    for (int i = 0; i < length; i++) {
      int sample = byteData.getInt16(i * 2, Endian.little);
      float32List[i] = sample / 32768.0;
    }
    return float32List;
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    _audioPlayer.dispose();
    _sttStream?.free();
    _sttRecognizer?.free();
    _ttsEngine?.free();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
          title: const Text('Jarvis AI (Official Engine)'),
          backgroundColor: Colors.blueAccent),
      body: Stack(
        children: [
          Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(15),
                  decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(10)),
                  child: Row(
                    children: [
                      const Icon(Icons.info, color: Colors.blueAccent),
                      const SizedBox(width: 10),
                      Expanded(
                          child: Text(_statusText,
                              style: const TextStyle(
                                  fontWeight: FontWeight.bold))),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
                Expanded(
                  child: TextField(
                    controller: _controller,
                    maxLines: null,
                    readOnly: true,
                    decoration: const InputDecoration(
                        border: OutlineInputBorder(), hintText: "Listening..."),
                  ),
                ),
                const SizedBox(height: 30),
                GestureDetector(
                  onTap: _isLoading ? null : _toggleRecording,
                  child: CircleAvatar(
                    radius: 40,
                    backgroundColor: _isLoading
                        ? Colors.grey
                        : (_isRecording ? Colors.red : Colors.blue),
                    child: Icon(_isRecording ? Icons.stop : Icons.mic,
                        color: Colors.white, size: 40),
                  ),
                ),
              ],
            ),
          ),
          if (_isLoading)
            Container(
              height: double.infinity,
              width: double.infinity,
              color: Colors.black.withOpacity(0.7),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(color: Colors.white),
                  const SizedBox(height: 20),
                  const Text("Setting up Jarvis...",
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold)),
                  const SizedBox(height: 10),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Text(_statusText,
                        textAlign: TextAlign.center,
                        style: const TextStyle(color: Colors.white70)),
                  ),
                  const SizedBox(height: 20),
                  const Text("(Using Official Asset Logic)",
                      style: TextStyle(color: Colors.white30, fontSize: 12)),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
