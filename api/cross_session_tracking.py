"""
Cross-Session Repetition Tracking for Cognitive Monitoring

Tracks concepts and questions across multiple recordings to detect:
- Repeated stories/statements across sessions
- Repeated questions across days
- Patterns of repetition over time

This is critical for dementia screening - someone repeating the same
story every visit is a stronger indicator than within-session repetition.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("speechscore.cross_session")

# Database connection helper
def _get_connection(db_path: str = None):
    """Get database connection - Supabase if available, else SQLite."""
    from speech_db import USE_SUPABASE, SUPABASE_AVAILABLE
    
    if USE_SUPABASE and SUPABASE_AVAILABLE:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            conn_str = os.getenv(
                "DATABASE_URL",
                "postgresql://postgres.fkxuqyvcvxklzrxjmzsa:Y4ZLP97tHSTQn7Jz@aws-1-us-east-1.pooler.supabase.com:6543/postgres"
            )
            return psycopg2.connect(conn_str, cursor_factory=RealDictCursor), 'postgres'
        except Exception as e:
            logger.warning(f"Supabase connection failed, falling back to SQLite: {e}")
    
    # SQLite fallback
    conn = sqlite3.connect(db_path or '/home/melchior/speech3/speechscore.db')
    conn.row_factory = sqlite3.Row
    return conn, 'sqlite'

# Lazy-loaded semantic model
_model = None

def get_model():
    """Lazy load sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded cross-session semantic model")
        except Exception as e:
            logger.warning(f"Failed to load semantic model: {e}")
            _model = False
    return _model if _model else None


def get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding for a text string."""
    model = get_model()
    if model is None:
        return None
    try:
        return model.encode(text, convert_to_numpy=True)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def get_user_concept_history(
    db_path: str,
    user_id: int,
    concept_type: Optional[str] = None,
    limit: int = 500
) -> List[Dict[str, Any]]:
    """
    Retrieve a user's concept history from the database.
    
    Args:
        db_path: Path to SQLite database
        user_id: User ID to look up
        concept_type: Filter by 'statement' or 'question' (None = all)
        limit: Maximum concepts to retrieve
    
    Returns:
        List of concept history records with embeddings
    """
    conn, db_type = _get_connection(db_path)
    
    try:
        if db_type == 'postgres':
            # PostgreSQL query
            query = """
                SELECT id, concept_text, concept_type, embedding, 
                       first_seen_at, last_seen_at, occurrence_count, speech_ids
                FROM user_concept_history
                WHERE user_id = %s
            """
            params = [user_id]
            
            if concept_type:
                query += " AND concept_type = %s"
                params.append(concept_type)
            
            query += " ORDER BY last_seen_at DESC LIMIT %s"
            params.append(limit)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
        else:
            # SQLite query
            query = """
                SELECT id, concept_text, concept_type, embedding, 
                       first_seen_at, last_seen_at, occurrence_count, speech_ids
                FROM user_concept_history
                WHERE user_id = ?
            """
            params = [user_id]
            
            if concept_type:
                query += " AND concept_type = ?"
                params.append(concept_type)
            
            query += " ORDER BY last_seen_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            record = dict(row)
            # Deserialize embedding
            if record.get('embedding'):
                if isinstance(record['embedding'], bytes):
                    record['embedding'] = np.frombuffer(record['embedding'], dtype=np.float32)
                elif isinstance(record['embedding'], memoryview):
                    record['embedding'] = np.frombuffer(bytes(record['embedding']), dtype=np.float32)
            record['speech_ids'] = json.loads(record.get('speech_ids') or '[]') if isinstance(record.get('speech_ids'), str) else (record.get('speech_ids') or [])
            results.append(record)
        
        return results
    finally:
        conn.close()


def store_concepts(
    db_path: str,
    user_id: int,
    speech_id: int,
    concepts: List[Dict[str, Any]],
    similarity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Store new concepts and update existing ones.
    
    If a concept is similar enough to an existing one, increment its count.
    Otherwise, create a new concept record.
    
    Args:
        db_path: Path to SQLite database
        user_id: User ID
        speech_id: Current speech/recording ID
        concepts: List of {text, type, embedding} dicts
        similarity_threshold: Similarity threshold for matching (0.75 = 75%)
    
    Returns:
        Summary of stored/updated concepts
    """
    conn = sqlite3.connect(db_path)
    
    # Get existing concepts for this user
    existing = get_user_concept_history(db_path, user_id)
    
    new_count = 0
    updated_count = 0
    
    for concept in concepts:
        text = concept['text']
        ctype = concept.get('type', 'statement')
        embedding = concept.get('embedding')
        
        if embedding is None:
            embedding = get_embedding(text)
        
        if embedding is None:
            continue
        
        # Check if similar concept exists
        matched = False
        for existing_concept in existing:
            if existing_concept['embedding'] is None:
                continue
            
            sim = cosine_similarity(embedding, existing_concept['embedding'])
            if sim >= similarity_threshold:
                # Update existing concept
                speech_ids = existing_concept['speech_ids']
                if speech_id not in speech_ids:
                    speech_ids.append(speech_id)
                
                conn.execute("""
                    UPDATE user_concept_history
                    SET last_seen_at = CURRENT_TIMESTAMP,
                        occurrence_count = occurrence_count + 1,
                        speech_ids = ?
                    WHERE id = ?
                """, (json.dumps(speech_ids), existing_concept['id']))
                
                updated_count += 1
                matched = True
                break
        
        if not matched:
            # Insert new concept
            conn.execute("""
                INSERT INTO user_concept_history 
                (user_id, concept_text, concept_type, embedding, speech_ids)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                text,
                ctype,
                embedding.astype(np.float32).tobytes(),
                json.dumps([speech_id])
            ))
            new_count += 1
    
    conn.commit()
    conn.close()
    
    return {
        'new_concepts': new_count,
        'updated_concepts': updated_count,
        'total_processed': len(concepts)
    }


def detect_cross_session_repetitions(
    db_path: str,
    user_id: int,
    current_concepts: List[str],
    current_questions: List[str],
    similarity_threshold: float = 0.70,
    min_days_apart: float = 0.0  # Minimum days between sessions to flag
) -> Dict[str, Any]:
    """
    Detect repetitions from previous sessions.
    
    Args:
        db_path: Path to SQLite database
        user_id: User ID
        current_concepts: Concepts from current recording
        current_questions: Questions from current recording
        similarity_threshold: Similarity threshold for matching
        min_days_apart: Only flag if last occurrence was at least this many days ago
    
    Returns:
        Cross-session repetition analysis
    """
    model = get_model()
    if model is None:
        return {
            'enabled': False,
            'error': 'Semantic model not available'
        }
    
    # Get historical concepts
    history = get_user_concept_history(db_path, user_id)
    if not history:
        return {
            'enabled': True,
            'cross_session_repetitions': [],
            'total_historical_concepts': 0,
            'message': 'No historical data for this user yet'
        }
    
    # Separate historical statements and questions
    hist_statements = [h for h in history if h['concept_type'] == 'statement']
    hist_questions = [h for h in history if h['concept_type'] == 'question']
    
    # Encode current concepts
    all_current = current_concepts + current_questions
    if not all_current:
        return {
            'enabled': True,
            'cross_session_repetitions': [],
            'total_historical_concepts': len(history),
            'message': 'No concepts in current recording'
        }
    
    current_embeddings = model.encode(all_current, convert_to_numpy=True, show_progress_bar=False)
    
    repetitions = []
    now = datetime.utcnow()
    
    # Check concepts against historical statements
    for i, concept in enumerate(current_concepts):
        embedding = current_embeddings[i]
        
        for hist in hist_statements:
            if hist['embedding'] is None:
                continue
            
            sim = cosine_similarity(embedding, hist['embedding'])
            if sim >= similarity_threshold:
                # Check time gap
                last_seen = datetime.fromisoformat(hist['last_seen_at'].replace('Z', '+00:00').replace('+00:00', ''))
                days_apart = (now - last_seen).total_seconds() / 86400
                
                if days_apart >= min_days_apart:
                    repetitions.append({
                        'current_text': concept,
                        'historical_text': hist['concept_text'],
                        'type': 'statement',
                        'similarity': round(sim, 3),
                        'first_seen': hist['first_seen_at'],
                        'last_seen': hist['last_seen_at'],
                        'days_since_last': round(days_apart, 1),
                        'total_occurrences': hist['occurrence_count'] + 1,
                        'sessions_appeared_in': len(hist['speech_ids']) + 1,
                        'concern_level': _calculate_cross_session_concern(
                            sim, days_apart, hist['occurrence_count']
                        )
                    })
    
    # Check questions against historical questions
    for i, question in enumerate(current_questions):
        embedding = current_embeddings[len(current_concepts) + i]
        
        for hist in hist_questions:
            if hist['embedding'] is None:
                continue
            
            sim = cosine_similarity(embedding, hist['embedding'])
            if sim >= similarity_threshold:
                last_seen = datetime.fromisoformat(hist['last_seen_at'].replace('Z', '+00:00').replace('+00:00', ''))
                days_apart = (now - last_seen).total_seconds() / 86400
                
                if days_apart >= min_days_apart:
                    repetitions.append({
                        'current_text': question,
                        'historical_text': hist['concept_text'],
                        'type': 'question',
                        'similarity': round(sim, 3),
                        'first_seen': hist['first_seen_at'],
                        'last_seen': hist['last_seen_at'],
                        'days_since_last': round(days_apart, 1),
                        'total_occurrences': hist['occurrence_count'] + 1,
                        'sessions_appeared_in': len(hist['speech_ids']) + 1,
                        'concern_level': _calculate_cross_session_concern(
                            sim, days_apart, hist['occurrence_count']
                        )
                    })
    
    # Sort by concern level
    repetitions.sort(key=lambda x: x['concern_level'], reverse=True)
    
    # Calculate summary stats
    if repetitions:
        avg_concern = sum(r['concern_level'] for r in repetitions) / len(repetitions)
        max_concern = max(r['concern_level'] for r in repetitions)
    else:
        avg_concern = 0.0
        max_concern = 0.0
    
    return {
        'enabled': True,
        'cross_session_repetitions': repetitions[:20],  # Top 20
        'total_cross_session_matches': len(repetitions),
        'average_concern_level': round(avg_concern, 3),
        'max_concern_level': round(max_concern, 3),
        'total_historical_concepts': len(history),
        'historical_statements': len(hist_statements),
        'historical_questions': len(hist_questions)
    }


def _calculate_cross_session_concern(
    similarity: float,
    days_apart: float,
    previous_occurrences: int
) -> float:
    """
    Calculate concern level for cross-session repetition.
    
    Factors:
    - Higher similarity = more concern
    - More days apart = more concern (forgetting but repeating)
    - More previous occurrences = more concern (pattern)
    
    Returns: 0.0-1.0 concern score
    """
    # Base: similarity contributes 40%
    base = similarity * 0.4
    
    # Time factor: 0-1 based on days (30+ days = max concern)
    if days_apart < 1:
        time_factor = 0.2  # Same day, less concerning
    elif days_apart < 7:
        time_factor = 0.5  # Within a week
    elif days_apart < 30:
        time_factor = 0.8  # Within a month
    else:
        time_factor = 1.0  # Over a month
    
    # Occurrence factor: more occurrences = more concern
    # Caps at 5+ occurrences
    occurrence_factor = min(previous_occurrences / 5.0, 1.0)
    
    # Combine: 40% similarity + 35% time + 25% occurrences
    concern = base + (time_factor * 0.35) + (occurrence_factor * 0.25)
    
    return round(min(concern, 1.0), 3)


def analyze_metric_degradation(
    db_path: str,
    user_id: int,
    profile_id: Optional[int] = None,
    current_metrics: Dict[str, float] = None,
    lookback_days: int = 90
) -> Dict[str, Any]:
    """
    Analyze cognitive metric degradation over time.
    
    Tracks key metrics across sessions to detect declining trends:
    - Speech tempo (WPM) - declining is concerning
    - Pause-per-utterance ratio - increasing is concerning
    - Hesitation rate - increasing is concerning
    - Repetition frequency - increasing is concerning
    
    Args:
        db_path: Path to database
        user_id: User ID
        profile_id: Optional profile ID for multi-profile support
        current_metrics: Current session metrics to include in analysis
        lookback_days: How many days back to analyze
    
    Returns:
        Degradation analysis with trends and risk assessment
    """
    conn, db_type = _get_connection(db_path)
    
    try:
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
        
        # Query historical metrics from speeches table
        if db_type == 'postgres':
            query = """
                SELECT 
                    s.id, s.created_at, s.duration,
                    s.audio_analysis, s.dementia_analysis
                FROM speeches s
                WHERE s.user_id = %s 
                  AND s.created_at > %s
                  AND s.scoring_profile = 'dementia'
            """
            params = [user_id, cutoff]
            
            if profile_id is not None:
                query += " AND s.profile_id = %s"
                params.append(profile_id)
            
            query += " ORDER BY s.created_at ASC"
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()
        else:
            query = """
                SELECT 
                    id, created_at, duration,
                    audio_analysis, dementia_analysis
                FROM speeches
                WHERE user_id = ? 
                  AND created_at > ?
                  AND scoring_profile = 'dementia'
            """
            params = [user_id, cutoff]
            
            if profile_id is not None:
                query += " AND profile_id = ?"
                params.append(profile_id)
            
            query += " ORDER BY created_at ASC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        # Extract metrics from each session
        sessions = []
        for row in rows:
            record = dict(row)
            
            # Parse JSON fields
            audio = record.get('audio_analysis')
            dementia = record.get('dementia_analysis')
            
            if isinstance(audio, str):
                try:
                    audio = json.loads(audio)
                except:
                    audio = {}
            if isinstance(dementia, str):
                try:
                    dementia = json.loads(dementia)
                except:
                    dementia = {}
            
            # Extract metrics
            wpm = 0
            if audio and 'speaking_rate' in audio:
                wpm = audio['speaking_rate'].get('overall_wpm', 0)
            
            pause_ratio = 0
            hesitation_rate = 0
            repetition_count = 0
            
            if dementia:
                pause_metrics = dementia.get('pause_metrics', {})
                pause_ratio = pause_metrics.get('pause_per_utterance_ratio', 0)
                
                # Sum from all speakers
                for speaker_data in dementia.get('speakers', {}).values():
                    metrics = speaker_data.get('metrics', {})
                    hesitation_rate += metrics.get('aphasic_events', 0)
                    repetition_count += metrics.get('repeated_concepts', 0)
                    repetition_count += metrics.get('repeated_questions', 0)
            
            sessions.append({
                'date': record['created_at'],
                'speech_id': record['id'],
                'metrics': {
                    'speech_tempo': wpm,
                    'pause_per_utterance': pause_ratio,
                    'hesitation_rate': hesitation_rate,
                    'repetition_count': repetition_count
                }
            })
        
        # Add current metrics if provided
        if current_metrics:
            sessions.append({
                'date': datetime.utcnow().isoformat(),
                'speech_id': None,
                'metrics': current_metrics
            })
        
        if len(sessions) < 3:
            return {
                'enabled': True,
                'sessions_analyzed': len(sessions),
                'message': 'Insufficient history for trend analysis (need 3+ sessions)',
                'overall_trend': 'unknown'
            }
        
        # Calculate trends for each metric
        def calculate_trend(values: List[float]) -> Dict[str, Any]:
            """Calculate linear trend and statistics."""
            if not values or len(values) < 2:
                return {'trend': 0, 'change_pct': 0, 'direction': 'stable'}
            
            # Simple linear regression
            n = len(values)
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            
            # Percentage change from first to last
            if values[0] != 0:
                change_pct = ((values[-1] - values[0]) / values[0]) * 100
            else:
                change_pct = 0
            
            # Determine direction
            if abs(change_pct) < 5:
                direction = 'stable'
            elif change_pct > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
            
            return {
                'slope': round(slope, 4),
                'change_pct': round(change_pct, 1),
                'direction': direction,
                'first_value': round(values[0], 2),
                'last_value': round(values[-1], 2),
                'min': round(min(values), 2),
                'max': round(max(values), 2)
            }
        
        # Extract metric series
        speech_tempo_series = [s['metrics']['speech_tempo'] for s in sessions if s['metrics']['speech_tempo'] > 0]
        pause_ratio_series = [s['metrics']['pause_per_utterance'] for s in sessions]
        hesitation_series = [s['metrics']['hesitation_rate'] for s in sessions]
        repetition_series = [s['metrics']['repetition_count'] for s in sessions]
        
        trends = {
            'speech_tempo': calculate_trend(speech_tempo_series),
            'pause_per_utterance': calculate_trend(pause_ratio_series),
            'hesitation_rate': calculate_trend(hesitation_series),
            'repetition_count': calculate_trend(repetition_series)
        }
        
        # Calculate overall degradation score
        # Negative for speech tempo (slowing bad), positive for others (increasing bad)
        degradation_score = 0
        concerning_metrics = []
        
        if trends['speech_tempo']['direction'] == 'decreasing' and trends['speech_tempo']['change_pct'] < -10:
            degradation_score += abs(trends['speech_tempo']['change_pct']) / 100
            concerning_metrics.append(f"Speech tempo declining {abs(trends['speech_tempo']['change_pct']):.0f}%")
        
        if trends['pause_per_utterance']['direction'] == 'increasing' and trends['pause_per_utterance']['change_pct'] > 20:
            degradation_score += trends['pause_per_utterance']['change_pct'] / 100
            concerning_metrics.append(f"Pause ratio increasing {trends['pause_per_utterance']['change_pct']:.0f}%")
        
        if trends['hesitation_rate']['direction'] == 'increasing' and trends['hesitation_rate']['change_pct'] > 20:
            degradation_score += trends['hesitation_rate']['change_pct'] / 200
            concerning_metrics.append(f"Hesitations increasing {trends['hesitation_rate']['change_pct']:.0f}%")
        
        if trends['repetition_count']['direction'] == 'increasing' and trends['repetition_count']['change_pct'] > 30:
            degradation_score += trends['repetition_count']['change_pct'] / 200
            concerning_metrics.append(f"Repetitions increasing {trends['repetition_count']['change_pct']:.0f}%")
        
        # Overall trend assessment
        if degradation_score >= 0.5:
            overall_trend = 'declining'
            risk_message = 'Multiple metrics showing decline - recommend clinical evaluation'
        elif degradation_score >= 0.2:
            overall_trend = 'mild_decline'
            risk_message = 'Some metrics showing slight decline - continue monitoring'
        elif degradation_score < 0:
            overall_trend = 'improving'
            risk_message = 'Metrics stable or improving'
        else:
            overall_trend = 'stable'
            risk_message = 'Metrics within normal variation'
        
        return {
            'enabled': True,
            'sessions_analyzed': len(sessions),
            'period_days': lookback_days,
            'trends': trends,
            'degradation_score': round(degradation_score, 3),
            'overall_trend': overall_trend,
            'risk_message': risk_message,
            'concerning_metrics': concerning_metrics,
            'session_dates': [s['date'] for s in sessions]
        }
        
    except Exception as e:
        logger.error(f"Degradation analysis error: {e}")
        return {
            'enabled': False,
            'error': str(e)
        }
    finally:
        conn.close()


def get_user_repetition_summary(
    db_path: str,
    user_id: int,
    days: int = 30
) -> Dict[str, Any]:
    """
    Get a summary of a user's repetition patterns over time.
    
    Args:
        db_path: Path to SQLite database
        user_id: User ID
        days: Look back this many days
    
    Returns:
        Summary of repetition patterns
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    # Get concepts with multiple occurrences
    cursor = conn.execute("""
        SELECT concept_text, concept_type, occurrence_count, 
               first_seen_at, last_seen_at, speech_ids
        FROM user_concept_history
        WHERE user_id = ? AND occurrence_count > 1 AND last_seen_at > ?
        ORDER BY occurrence_count DESC
        LIMIT 20
    """, (user_id, cutoff))
    
    frequent_concepts = []
    for row in cursor:
        speech_ids = json.loads(row['speech_ids'] or '[]')
        frequent_concepts.append({
            'text': row['concept_text'],
            'type': row['concept_type'],
            'occurrences': row['occurrence_count'],
            'sessions': len(speech_ids),
            'first_seen': row['first_seen_at'],
            'last_seen': row['last_seen_at']
        })
    
    # Get total stats
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total_concepts,
            SUM(CASE WHEN occurrence_count > 1 THEN 1 ELSE 0 END) as repeated_concepts,
            AVG(occurrence_count) as avg_occurrences,
            MAX(occurrence_count) as max_occurrences
        FROM user_concept_history
        WHERE user_id = ? AND last_seen_at > ?
    """, (user_id, cutoff))
    
    stats = dict(cursor.fetchone())
    conn.close()
    
    return {
        'period_days': days,
        'total_unique_concepts': stats['total_concepts'] or 0,
        'concepts_repeated': stats['repeated_concepts'] or 0,
        'average_occurrences': round(stats['avg_occurrences'] or 0, 2),
        'max_occurrences': stats['max_occurrences'] or 0,
        'most_repeated': frequent_concepts
    }
