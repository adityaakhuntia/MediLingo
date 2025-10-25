import os
import sqlite3
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import speech_recognition as sr
from pydub import AudioSegment
import openai
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'temp_audio'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# OpenAI configuration (optional - for better transcription)
# Uncomment and add your API key for enhanced transcription
# openai.api_key = "your-openai-api-key-here"

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class DatabaseManager:
    def __init__(self, db_path='voice_transcriptions.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                language VARCHAR(10) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                audio_duration REAL,
                confidence_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcription_id INTEGER,
                filename VARCHAR(255),
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_transcription(self, text, language, audio_duration=None, confidence=None):
        """Save a transcription to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transcriptions (text, language, audio_duration, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (text, language, audio_duration, confidence))
        
        transcription_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return transcription_id
    
    def get_transcriptions(self, limit=50):
        """Retrieve recent transcriptions from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, text, language, created_at, audio_duration, confidence_score
            FROM transcriptions
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        transcriptions = []
        for row in rows:
            transcriptions.append({
                'id': row[0],
                'text': row[1],
                'language': row[2],
                'created_at': row[3],
                'audio_duration': row[4],
                'confidence_score': row[5]
            })
        
        return transcriptions
    
    def clear_transcriptions(self):
        """Clear all transcriptions from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM transcriptions')
        cursor.execute('DELETE FROM audio_files')
        
        conn.commit()
        conn.close()

class AudioTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
    
    def transcribe_with_speech_recognition(self, audio_file_path, language='en-US'):
        """Transcribe audio using Google Speech Recognition API."""
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = self.recognizer.record(source)
            
            # Convert language code format
            lang_code = self.convert_language_code(language)
            
            # Perform transcription
            text = self.recognizer.recognize_google(audio_data, language=lang_code)
            return {
                'success': True,
                'text': text,
                'confidence': 0.8,  # Google API doesn't provide confidence scores
                'method': 'google_speech_recognition'
            }
        
        except sr.UnknownValueError:
            return {
                'success': False,
                'error': 'Could not understand audio',
                'method': 'google_speech_recognition'
            }
        except sr.RequestError as e:
            return {
                'success': False,
                'error': f'Error with speech recognition service: {e}',
                'method': 'google_speech_recognition'
            }
    
    def transcribe_with_openai(self, audio_file_path, language='en'):
        """Transcribe audio using OpenAI Whisper API (requires API key)."""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    "whisper-1",
                    audio_file,
                    language=self.get_whisper_language_code(language)
                )
            
            return {
                'success': True,
                'text': transcript['text'],
                'confidence': 0.95,  # Whisper generally has high accuracy
                'method': 'openai_whisper'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'OpenAI transcription error: {e}',
                'method': 'openai_whisper'
            }
    
    def convert_language_code(self, language_code):
        """Convert language codes to the format expected by Google Speech Recognition."""
        language_mapping = {
            'en-US': 'en-US',
            'hi-IN': 'hi-IN',  # Hindi (India)
            'ta-IN': 'ta-IN',  # Tamil (India)
            'es-ES': 'es-ES',
            'fr-FR': 'fr-FR',
            'de-DE': 'de-DE',
            'it-IT': 'it-IT',
            'pt-BR': 'pt-BR',
            'ru-RU': 'ru-RU',
            'ja-JP': 'ja-JP',
            'ko-KR': 'ko-KR',
            'zh-CN': 'zh-CN'
        }
        return language_mapping.get(language_code, 'en-US')
    
    def get_whisper_language_code(self, language_code):
        """Convert language codes to the format expected by OpenAI Whisper."""
        whisper_language_mapping = {
            'en-US': 'en',
            'hi-IN': 'hi',  # Hindi
            'ta-IN': 'ta',  # Tamil
            'es-ES': 'es',
            'fr-FR': 'fr',
            'de-DE': 'de',
            'it-IT': 'it',
            'pt-BR': 'pt',
            'ru-RU': 'ru',
            'ja-JP': 'ja',
            'ko-KR': 'ko',
            'zh-CN': 'zh'
        }
        return whisper_language_mapping.get(language_code, 'en')
    
    def get_language_name(self, language_code):
        """Get human-readable language name."""
        language_names = {
            'en-US': 'English (US)',
            'hi-IN': 'हिंदी Hindi',
            'ta-IN': 'தமிழ் Tamil',
            'es-ES': 'Spanish',
            'fr-FR': 'French',
            'de-DE': 'German',
            'it-IT': 'Italian',
            'pt-BR': 'Portuguese',
            'ru-RU': 'Russian',
            'ja-JP': 'Japanese',
            'ko-KR': 'Korean',
            'zh-CN': 'Chinese'
        }
        return language_names.get(language_code, 'Unknown')
    
    def convert_audio_format(self, input_file_path, output_file_path):
        """Convert audio to WAV format for better compatibility."""
        try:
            audio = AudioSegment.from_file(input_file_path)
            audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
            audio.export(output_file_path, format="wav")
            return True
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return False

# Initialize components
db_manager = DatabaseManager()
transcriber = AudioTranscriber()

@app.route('/')
def index():
    """Serve the main application page."""
    return jsonify({
        'message': 'Voice-to-Text AI API is running with Hindi and Tamil support!',
        'supported_languages': {
            'en-US': 'English (US)',
            'hi-IN': 'हिंदी Hindi',
            'ta-IN': 'தமிழ் Tamil',
            'es-ES': 'Spanish',
            'fr-FR': 'French',
            'de-DE': 'German',
            'it-IT': 'Italian',
            'pt-BR': 'Portuguese',
            'ja-JP': 'Japanese',
            'ko-KR': 'Korean',
            'zh-CN': 'Chinese'
        },
        'endpoints': {
            'transcribe': '/api/transcribe (POST)',
            'save_transcription': '/api/save_transcription (POST)',
            'get_transcriptions': '/api/get_transcriptions (GET)',
            'clear_transcriptions': '/api/clear_transcriptions (DELETE)',
            'supported_languages': '/api/languages (GET)'
        }
    })

@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Get list of supported languages."""
    return jsonify({
        'languages': [
            {'code': 'en-US', 'name': 'English (US)', 'native_name': 'English', 'flag': '🇺🇸'},
            {'code': 'hi-IN', 'name': 'Hindi', 'native_name': 'हिंदी', 'flag': '🇮🇳'},
            {'code': 'ta-IN', 'name': 'Tamil', 'native_name': 'தமிழ்', 'flag': '🇮🇳'},
            {'code': 'es-ES', 'name': 'Spanish', 'native_name': 'Español', 'flag': '🇪🇸'},
            {'code': 'fr-FR', 'name': 'French', 'native_name': 'Français', 'flag': '🇫🇷'},
            {'code': 'de-DE', 'name': 'German', 'native_name': 'Deutsch', 'flag': '🇩🇪'},
            {'code': 'it-IT', 'name': 'Italian', 'native_name': 'Italiano', 'flag': '🇮🇹'},
            {'code': 'pt-BR', 'name': 'Portuguese', 'native_name': 'Português', 'flag': '🇧🇷'},
            {'code': 'ja-JP', 'name': 'Japanese', 'native_name': '日本語', 'flag': '🇯🇵'},
            {'code': 'ko-KR', 'name': 'Korean', 'native_name': '한국어', 'flag': '🇰🇷'},
            {'code': 'zh-CN', 'name': 'Chinese', 'native_name': '中文', 'flag': '🇨🇳'}
        ]
    })

@app.route('/api/transliterate', methods=['POST'])
def transliterate_text_endpoint():
    """Transliterate text from Indian languages to Roman script."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        language = data.get('language', 'en-US')
        
        if language not in ['hi-IN', 'ta-IN']:
            return jsonify({'error': 'Transliteration only supported for Hindi and Tamil'}), 400
        
        transliterated_text = transcriber.transliterate_text(text, language)
        
        return jsonify({
            'original_text': text,
            'transliterated_text': transliterated_text,
            'source_language': language,
            'target_script': 'Roman',
            'method': 'indic_transliteration' if TRANSLITERATION_AVAILABLE else 'basic_mapping'
        })
    
    except Exception as e:
        return jsonify({'error': f'Transliteration error: {str(e)}'}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Handle audio transcription requests."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', 'en-US')
    
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate language code
    supported_languages = ['en-US', 'hi-IN', 'ta-IN', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR', 'ja-JP', 'ko-KR', 'zh-CN']
    if language not in supported_languages:
        return jsonify({'error': f'Unsupported language: {language}. Supported: {supported_languages}'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        
        # Convert to WAV format if needed
        wav_path = temp_path.replace(os.path.splitext(temp_path)[1], '.wav')
        if not transcriber.convert_audio_format(temp_path, wav_path):
            return jsonify({'error': 'Audio format conversion failed'}), 500
        
        # Calculate audio duration
        try:
            audio = AudioSegment.from_wav(wav_path)
            duration = len(audio) / 1000.0  # Convert to seconds
        except:
            duration = None
        
        # Try OpenAI Whisper first (if API key is configured), then fall back to Google Speech Recognition
        result = None
        if hasattr(openai, 'api_key') and openai.api_key:
            result = transcriber.transcribe_with_openai(wav_path, language)
        
        if not result or not result['success']:
            result = transcriber.transcribe_with_speech_recognition(wav_path, language)
        
        # Clean up temporary files
        try:
            os.remove(temp_path)
            if os.path.exists(wav_path) and wav_path != temp_path:
                os.remove(wav_path)
        except:
            pass
        
        if result['success']:
            # Auto-save transcription
            transcription_id = db_manager.save_transcription(
                text=result['text'],
                language=language,
                audio_duration=duration,
                confidence=result.get('confidence', 0.8)
            )
            
            return jsonify({
                'transcription': result['text'],
                'confidence': result.get('confidence', 0.8),
                'duration': duration,
                'method': result['method'],
                'language': language,
                'language_name': transcriber.get_language_name(language),
                'transcription_id': transcription_id
            })
        else:
            return jsonify({'error': result['error']}), 400
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/api/save_transcription', methods=['POST'])
def save_transcription():
    """Save a transcription to the database."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No transcription text provided'}), 400
        
        text = data['text']
        language = data.get('language', 'en-US')
        audio_duration = data.get('duration')
        confidence = data.get('confidence')
        
        # Save to database
        transcription_id = db_manager.save_transcription(
            text=text,
            language=language,
            audio_duration=audio_duration,
            confidence=confidence
        )
        
        return jsonify({
            'success': True,
            'transcription_id': transcription_id,
            'message': f'Transcription saved successfully in {transcriber.get_language_name(language)}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Save error: {str(e)}'}), 500

@app.route('/api/get_transcriptions', methods=['GET'])
def get_transcriptions():
    """Retrieve transcriptions from the database."""
    try:
        limit = request.args.get('limit', 50, type=int)
        language_filter = request.args.get('language', None)
        
        transcriptions = db_manager.get_transcriptions(limit=limit)
        
        # Filter by language if specified
        if language_filter:
            transcriptions = [t for t in transcriptions if t['language'] == language_filter]
        
        # Add language names
        for transcription in transcriptions:
            transcription['language_name'] = transcriber.get_language_name(transcription['language'])
        
        return jsonify({
            'transcriptions': transcriptions,
            'total': len(transcriptions),
            'filter': language_filter
        })
    
    except Exception as e:
        return jsonify({'error': f'Retrieval error: {str(e)}'}), 500

@app.route('/api/clear_transcriptions', methods=['DELETE'])
def clear_transcriptions():
    """Clear all transcriptions from the database."""
    try:
        db_manager.clear_transcriptions()
        
        return jsonify({
            'success': True,
            'message': 'All transcriptions cleared successfully'
        })
    
    except Exception as e:
        return jsonify({'error': f'Clear error: {str(e)}'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about transcriptions."""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute('SELECT COUNT(*) FROM transcriptions')
        total_count = cursor.fetchone()[0]
        
        # Get language distribution
        cursor.execute('''
            SELECT language, COUNT(*) as count
            FROM transcriptions
            GROUP BY language
            ORDER BY count DESC
        ''')
        language_stats = cursor.fetchall()
        
        # Get average confidence score
        cursor.execute('SELECT AVG(confidence_score) FROM transcriptions WHERE confidence_score IS NOT NULL')
        avg_confidence = cursor.fetchone()[0]
        
        # Get total audio duration
        cursor.execute('SELECT SUM(audio_duration) FROM transcriptions WHERE audio_duration IS NOT NULL')
        total_duration = cursor.fetchone()[0]
        
        # Get Hindi and Tamil specific stats
        cursor.execute('SELECT COUNT(*) FROM transcriptions WHERE language IN ("hi-IN", "ta-IN")')
        indic_language_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Add language names to distribution
        language_distribution = []
        for lang, count in language_stats:
            language_distribution.append({
                'language': lang,
                'language_name': transcriber.get_language_name(lang),
                'count': count
            })
        
        return jsonify({
            'total_transcriptions': total_count,
            'language_distribution': language_distribution,
            'average_confidence': round(avg_confidence, 2) if avg_confidence else None,
            'total_audio_duration': round(total_duration, 2) if total_duration else None,
            'indic_language_transcriptions': indic_language_count,
            'supported_languages': len(['en-US', 'hi-IN', 'ta-IN', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR', 'ja-JP', 'ko-KR', 'zh-CN'])
        })
    
    except Exception as e:
        return jsonify({'error': f'Stats error: {str(e)}'}), 500

@app.route('/api/search', methods=['GET'])
def search_transcriptions():
    """Search transcriptions by text content."""
    try:
        query = request.args.get('q', '')
        language_filter = request.args.get('language', None)
        
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        if language_filter:
            cursor.execute('''
                SELECT id, text, language, created_at, audio_duration, confidence_score
                FROM transcriptions
                WHERE text LIKE ? AND language = ?
                ORDER BY created_at DESC
                LIMIT 20
            ''', (f'%{query}%', language_filter))
        else:
            cursor.execute('''
                SELECT id, text, language, created_at, audio_duration, confidence_score
                FROM transcriptions
                WHERE text LIKE ?
                ORDER BY created_at DESC
                LIMIT 20
            ''', (f'%{query}%',))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'text': row[1],
                'language': row[2],
                'language_name': transcriber.get_language_name(row[2]),
                'created_at': row[3],
                'audio_duration': row[4],
                'confidence_score': row[5]
            })
        
        return jsonify({
            'results': results, 
            'query': query,
            'language_filter': language_filter,
            'total_found': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': f'Search error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.1',
        'features': ['hindi_support', 'tamil_support', 'multi_language'],
        'supported_languages': ['en-US', 'hi-IN', 'ta-IN', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-BR', 'ja-JP', 'ko-KR', 'zh-CN']
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("Starting Voice-to-Text AI Server with Hindi and Tamil support...")
    print("Server will be available at: http://localhost:5000")
    print()
    print("Supported Languages:")
    print("  🇺🇸 English (US) - en-US")
    print("  🇮🇳 हिंदी Hindi - hi-IN")
    print("  🇮🇳 தமிழ் Tamil - ta-IN")
    print()
    print("API endpoints:")
    print("  - POST /api/transcribe - Upload audio for transcription")
    print("  - GET /api/languages - Get supported languages")
    print("  - POST /api/save_transcription - Save transcription to database")
    print("  - GET /api/get_transcriptions - Retrieve saved transcriptions")
    print("  - DELETE /api/clear_transcriptions - Clear all transcriptions")
    print("  - GET /api/stats - Get transcription statistics")
    print("  - GET /api/search - Search transcriptions")
    print("  - GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=5000)