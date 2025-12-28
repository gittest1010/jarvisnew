import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle, ByteData;
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
// OFFICIAL UTILS CODE (Integrated from utils.dart)
// ============================================================

// https://stackoverflow.com/questions/68862225/flutter-how-to-get-all-files-from-assets-folder-in-one-list
Future<List<String>> getAllAssetFiles() async {
  final AssetManifest assetManifest =
      await AssetManifest.loadFromAssetBundle(rootBundle);
  final List<String> assets = assetManifest.listAssets();
  return assets;
}

String stripLeadingDirectory(String src, {int n = 1}) {
  return p.joinAll(p.split(src).sublist(n));
}

Future<void> copyAllAssetFiles() async {
  final allFiles = await getAllAssetFiles();
  for (final src in allFiles) {
    // Note: We might need to adjust stripping based on your asset structure
    // If assets are in 'assets/models/file', strip 1 makes it 'models/file'
    // If assets are just 'assets/file', strip 1 makes it 'file'
    final dst = stripLeadingDirectory(src);
    await copyAssetFile(src, dst);
  }
}

// Copy the asset file from src to dst.
// If dst already exists, then just skip the copy
Future<String> copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationSupportDirectory();
  if (dst == null) {
    dst = p.basename(src);
  }
  final target = p.join(directory.path, dst);
  bool exists = await File(target).exists();

  final data = await rootBundle.load(src);

  // Only copy if size differs or doesn't exist
  if (!exists || File(target).lengthSync() != data.lengthInBytes) {
    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await (await File(target).create(recursive: true)).writeAsBytes(bytes);
    debugPrint("Copied asset: $src -> $target");
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
// OFFICIAL MODEL CREATION CODE (Integrated from model.dart)
// ============================================================

Future<sherpa_onnx.OfflineTts> createOfficialOfflineTts() async {
  // 1. Copy all assets first (Using official utility)
  // This ensures .onnx, .json, and .tar.bz2 are in AppSupportDirectory
  await copyAllAssetFiles();

  // 2. Initialize Bindings
  sherpa_onnx.initBindings();

  final Directory directory = await getApplicationSupportDirectory();

  // 3. Define Model Paths (Adapted for your Hindi Model)
  String modelName = 'hi_IN-pratham-medium.onnx';
  String tokens = 'hi_IN-pratham-medium.onnx.json';
  String dataDir = 'espeak-ng-data'; // Expected extracted folder

  // 4. Handle Zip Extraction for espeak-ng-data
  // Official code usually assumes folder exists or provides script.
  // We must extract the zip here because we are on Android.
  final zipFile = File(p.join(directory.path, 'espeak-ng-data.tar.bz2'));
  final targetDir = Directory(p.join(directory.path, dataDir));

  if (!await targetDir.exists()) {
    if (await zipFile.exists()) {
      debugPrint("Extracting espeak-ng-data.tar.bz2...");
      // Using compute to avoid UI freeze, same as your old robust code
      final bytes = await zipFile.readAsBytes();
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

  // 5. Construct Full Paths
  modelName = p.join(directory.path, modelName);
  tokens = p.join(directory.path,
      tokens); // Note: Json is often used as tokens or lexicon config
  dataDir = p.join(directory.path, dataDir);

  // 6. Create Configuration (Official Structure)
  final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
    model: modelName,
    lexicon: '', // Hindi model usually uses onnx.json or internal
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

// Background Isolate for Zip (Helper)
Archive _decodeArchiveInIsolate(List<int> bytes) {
  final decoded = BZip2Decoder().decodeBytes(bytes);
  return TarDecoder().decodeBytes(decoded);
}

// ============================================================
// MAIN APP (Your Jarvis UI)
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

      // A. Initialize TTS using the OFFICIAL logic
      // This handles initBindings, copying assets, and creating the engine
      _ttsEngine = await createOfficialOfflineTts();

      setState(() => _statusText = "Initializing STT...");

      // B. Initialize STT (Online)
      // We need to get paths for STT models. Since copyAllAssetFiles()
      // put them in ApplicationSupportDirectory, we access them there.
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

      // Use the utility function to generate filename
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

  // --- HELPER: Bytes to Float32 ---
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
