import json
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, jsonify, redirect, url_for
from flask_login import login_required, current_user
from database.connection import db
from models.interview import InterviewSession
from models.response import Response
from services.ml_inference_service import MLInferenceService

SKLEARN_AVAILABLE = None
ml_classifier = None

dashboard_bp = Blueprint('dashboard', __name__)

def train_readiness_classifier():
    """
    Creates a synthetic dataset of mock interview scores and trains a 
    Decision Tree Classifier to predict placement readiness categories.
    """
    global SKLEARN_AVAILABLE, ml_classifier
    if ml_classifier is not None:
        return ml_classifier
    try:
        from sklearn.tree import DecisionTreeClassifier
        SKLEARN_AVAILABLE = True
    except Exception as e:
        SKLEARN_AVAILABLE = False
        print(f"scikit-learn not loaded: {e}. Dashboard ML features will run in fallback mode.")
        return None
        
    # Columns: Comm, Tech, Quality, Prof, Confidence (Gaze/Speech)
    # Target: 0 (Needs Focus), 1 (Needs Practice), 2 (Placement Ready)
    training_data = [
        # Placement Ready
        [90, 85, 90, 90, 95, 2],
        [85, 80, 85, 85, 90, 2],
        [80, 85, 80, 90, 80, 2],
        [95, 75, 85, 85, 85, 2],
        [80, 80, 80, 80, 80, 2],
        
        # Needs Practice
        [70, 75, 70, 70, 75, 1],
        [65, 70, 65, 75, 70, 1],
        [80, 60, 70, 65, 75, 1],
        [60, 80, 65, 70, 60, 1],
        [75, 65, 75, 70, 65, 1],
        
        # Needs Focus
        [50, 50, 45, 55, 50, 0],
        [40, 55, 50, 40, 45, 0],
        [60, 40, 45, 50, 55, 0],
        [30, 45, 40, 40, 30, 0],
        [50, 35, 30, 45, 40, 0]
    ]
    
    # Generate some minor random variance to expand the dataset to 100 samples
    extended_data = []
    for row in training_data:
        extended_data.append(row)
        for _ in range(5):
            noise = np.random.randint(-4, 5, size=5)
            new_features = np.clip(np.array(row[:5]) + noise, 0, 100).tolist()
            extended_data.append(new_features + [row[5]])
            
    df = pd.DataFrame(extended_data, columns=['comm', 'tech', 'quality', 'prof', 'conf', 'label'])
    
    X = df[['comm', 'tech', 'quality', 'prof', 'conf']]
    y = df['label']
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf

# Readiness model is initialized lazily when requested via train_readiness_classifier()

@dashboard_bp.route('/')
def index():
    """Landing Page (visible without authentication)"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.home'))
    return render_template('index.html')

@dashboard_bp.route('/dashboard')
@login_required
def home():
    """Main authenticated student dashboard"""
    # Fetch all completed sessions
    sessions = db.session.query(InterviewSession).filter_by(user_id=current_user.id, status='completed').order_by(InterviewSession.created_at.desc()).all()
    
    total_interviews = len(sessions)
    
    if total_interviews == 0:
        return render_template('dashboard.html', total_interviews=0, sessions=[], ml_inflation={"available": False, "predictions": [], "message": "Complete a resume-based interview to see an optional ML comparison."})
        
    # Averages
    avg_overall = round(sum(s.overall_score for s in sessions) / total_interviews, 1)
    avg_comm = round(sum(s.communication_score for s in sessions) / total_interviews, 1)
    avg_tech = round(sum(s.technical_score for s in sessions) / total_interviews, 1)
    avg_quality = round(sum(s.answer_quality_score for s in sessions) / total_interviews, 1)
    avg_prof = round(sum(s.professionalism_score for s in sessions) / total_interviews, 1)
    avg_conf = round(sum(s.confidence_score for s in sessions) / total_interviews, 1)
    
    # Classification of skills: Strong, Moderate, Weak
    skill_scores = {
        "Technical Knowledge": avg_tech,
        "Communication Fluency": avg_comm,
        "Answer Quality": avg_quality,
        "Professional Presence": avg_prof,
        "Confidence & Attention": avg_conf
    }
    
    skills_classified = {"strong": [], "moderate": [], "weak": []}
    for name, score in skill_scores.items():
        if score >= 75.0:
            skills_classified["strong"].append((name, score))
        elif 55.0 <= score < 75.0:
            skills_classified["moderate"].append((name, score))
        else:
            skills_classified["weak"].append((name, score))
            
    # ML Prediction for Placement Readiness
    readiness_label = "Needs Practice"
    readiness_class = "warning"
    readiness_desc = "You are making steady progress. Focus on your weaker skills to move to the next level."
    
    if SKLEARN_AVAILABLE and ml_classifier:
        features = np.array([[avg_comm, avg_tech, avg_quality, avg_prof, avg_conf]])
        pred = ml_classifier.predict(features)[0]
        if pred == 2:
            readiness_label = "Placement Ready"
            readiness_class = "success"
            readiness_desc = "Excellent performance! You demonstrate the required skills to clear real corporate interviews."
        elif pred == 0:
            readiness_label = "Needs Focus"
            readiness_class = "danger"
            readiness_desc = "You require active preparation. Focus on foundational technical answers and reduce filler words."
    else:
        # Fallback heuristic
        if avg_overall >= 75:
            readiness_label = "Placement Ready"
            readiness_class = "success"
            readiness_desc = "Excellent performance! You demonstrate the required skills to clear real corporate interviews."
        elif avg_overall < 55:
            readiness_label = "Needs Focus"
            readiness_class = "danger"
            readiness_desc = "You require active preparation. Focus on foundational technical answers and reduce filler words."
            
    # Keep the existing rule-based resume score and show the external ML result separately.
    ml_inflation = {"available": False, "predictions": [],
                    "message": "No matched resume claim data is available for ML inference."}
    for completed_session in sessions:
        try:
            competency_map = completed_session.get_competency_map()
            rule_analysis = competency_map.get("resume_inflation_analysis", {})
            mismatches = rule_analysis.get("skills_mismatch", [])
            if mismatches:
                ml_inflation = MLInferenceService().predict_skill_mismatches(mismatches)
                ml_inflation["rule_based_score"] = rule_analysis.get("resume_inflation_score")
                break
        except Exception as exc:
            ml_inflation = {"available": False, "predictions": [],
                            "message": f"ML comparison unavailable ({exc.__class__.__name__})."}
            break

    # Retrieve all responses for filler word summary
    responses = db.session.query(Response).join(InterviewSession).filter(InterviewSession.user_id == current_user.id).all()
    
    # Aggregated Filler Words
    agg_fillers = {}
    total_fillers = 0
    total_wpm = 0
    response_count_with_wpm = 0
    
    for r in responses:
        if r.wpm > 0:
            total_wpm += r.wpm
            response_count_with_wpm += 1
        total_fillers += r.filler_count
        fillers_dict = r.get_filler_words()
        for k, v in fillers_dict.items():
            agg_fillers[k] = agg_fillers.get(k, 0) + v
            
    avg_wpm = int(total_wpm / response_count_with_wpm) if response_count_with_wpm > 0 else 0
    sorted_fillers = sorted(agg_fillers.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Format sessions for dashboard listing (limit to 5)
    recent_sessions = sessions[:5]
    
    return render_template(
        'dashboard.html',
        total_interviews=total_interviews,
        avg_overall=avg_overall,
        avg_comm=avg_comm,
        avg_tech=avg_tech,
        avg_quality=avg_quality,
        avg_prof=avg_prof,
        avg_conf=avg_conf,
        skills=skills_classified,
        readiness_label=readiness_label,
        readiness_class=readiness_class,
        readiness_desc=readiness_desc,
        ml_inflation=ml_inflation,
        avg_wpm=avg_wpm,
        total_fillers=total_fillers,
        top_fillers=sorted_fillers,
        recent_sessions=recent_sessions,
        sessions=sessions
    )

@dashboard_bp.route('/api/chart-data')
@login_required
def chart_data():
    """API Endpoint that returns session history scores and speech metrics for Chart.js"""
    sessions = db.session.query(InterviewSession).filter_by(user_id=current_user.id, status='completed').order_by(InterviewSession.created_at.asc()).all()
    
    dates = []
    overall_scores = []
    comm_scores = []
    tech_scores = []
    
    for s in sessions:
        dates.append(s.created_at.strftime('%b %d, %H:%M'))
        overall_scores.append(round(s.overall_score, 1))
        comm_scores.append(round(s.communication_score, 1))
        tech_scores.append(round(s.technical_score, 1))
        
    return jsonify({
        "labels": dates,
        "overall": overall_scores,
        "communication": comm_scores,
        "technical": tech_scores
    })

@dashboard_bp.route('/history')
@login_required
def history():
    """Renders the detailed history of all interviews."""
    sessions = db.session.query(InterviewSession).filter_by(user_id=current_user.id).order_by(InterviewSession.created_at.desc()).all()
    return render_template('history.html', sessions=sessions)
