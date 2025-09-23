-- VoiceStand Learning System Database Schema
-- Replaces in-memory storage with persistent PostgreSQL storage

-- Enable pgvector extension for audio feature storage
CREATE EXTENSION IF NOT EXISTS vector;

-- Table for storing recognition history with audio features
CREATE TABLE IF NOT EXISTS recognition_history (
    id BIGSERIAL PRIMARY KEY,
    recognition_id VARCHAR(100) UNIQUE NOT NULL,
    audio_features VECTOR(1024), -- Audio features as vector for similarity matching
    recognized_text TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_used VARCHAR(100) NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    speaker_id VARCHAR(50),
    is_uk_english BOOLEAN DEFAULT FALSE,
    ground_truth TEXT, -- For training examples
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookups and similarity searches
CREATE INDEX IF NOT EXISTS idx_recognition_history_model ON recognition_history(model_used);
CREATE INDEX IF NOT EXISTS idx_recognition_history_uk ON recognition_history(is_uk_english);
CREATE INDEX IF NOT EXISTS idx_recognition_history_created ON recognition_history(created_at);
CREATE INDEX IF NOT EXISTS idx_recognition_history_features ON recognition_history USING ivfflat (audio_features vector_cosine_ops);

-- Table for storing learning patterns and UK English vocabulary mappings
CREATE TABLE IF NOT EXISTS learning_patterns (
    id BIGSERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL, -- 'uk_vocabulary', 'pronunciation', 'grammar', etc.
    source_text VARCHAR(500) NOT NULL,
    target_text VARCHAR(500) NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    usage_count INTEGER DEFAULT 1,
    accuracy_improvement REAL DEFAULT 0.0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for pattern lookups
CREATE INDEX IF NOT EXISTS idx_learning_patterns_type ON learning_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_source ON learning_patterns(source_text);
CREATE INDEX IF NOT EXISTS idx_learning_patterns_active ON learning_patterns(is_active);

-- Table for model performance metrics over time
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    accuracy REAL NOT NULL CHECK (accuracy >= 0 AND accuracy <= 100),
    uk_accuracy REAL CHECK (uk_accuracy >= 0 AND uk_accuracy <= 100),
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    sample_count INTEGER DEFAULT 0,
    total_processing_time_ms BIGINT DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Unique index to ensure one record per model
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_performance_unique ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_updated ON model_performance(last_updated);

-- Table for storing training examples with ground truth data
CREATE TABLE IF NOT EXISTS training_examples (
    id BIGSERIAL PRIMARY KEY,
    recognition_id VARCHAR(100) REFERENCES recognition_history(recognition_id),
    audio_features VECTOR(1024),
    ground_truth TEXT NOT NULL,
    predicted_text TEXT NOT NULL,
    accuracy_score REAL NOT NULL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
    model_used VARCHAR(100) NOT NULL,
    is_uk_english BOOLEAN DEFAULT FALSE,
    speaker_id VARCHAR(50),
    domain VARCHAR(50), -- Technical, Medical, Legal, Business, Academic, General
    correction_applied BOOLEAN DEFAULT FALSE,
    used_for_training BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for training data queries
CREATE INDEX IF NOT EXISTS idx_training_examples_model ON training_examples(model_used);
CREATE INDEX IF NOT EXISTS idx_training_examples_uk ON training_examples(is_uk_english);
CREATE INDEX IF NOT EXISTS idx_training_examples_domain ON training_examples(domain);
CREATE INDEX IF NOT EXISTS idx_training_examples_features ON training_examples USING ivfflat (audio_features vector_cosine_ops);

-- Table for overall system metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for metrics queries
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Table for recent activity log
CREATE TABLE IF NOT EXISTS activity_log (
    id BIGSERIAL PRIMARY KEY,
    activity_type VARCHAR(50) NOT NULL, -- 'uk_pattern', 'optimization', 'system', 'training'
    message TEXT NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for activity queries
CREATE INDEX IF NOT EXISTS idx_activity_log_type ON activity_log(activity_type);
CREATE INDEX IF NOT EXISTS idx_activity_log_timestamp ON activity_log(timestamp);

-- Table for learning insights and recommendations
CREATE TABLE IF NOT EXISTS learning_insights (
    id BIGSERIAL PRIMARY KEY,
    insight_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    recommendations JSONB NOT NULL,
    requires_retraining BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Index for insights queries
CREATE INDEX IF NOT EXISTS idx_learning_insights_type ON learning_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_learning_insights_active ON learning_insights(is_active);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_recognition_history_updated_at BEFORE UPDATE ON recognition_history FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_learning_patterns_updated_at BEFORE UPDATE ON learning_patterns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_model_performance_updated_at BEFORE UPDATE ON model_performance FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial model performance data
INSERT INTO model_performance (model_name, accuracy, uk_accuracy, weight, sample_count) VALUES
('whisper_small', 82.0, 84.0, 0.25, 1000),
('whisper_medium', 87.0, 89.0, 0.35, 1500),
('whisper_large', 92.0, 94.0, 0.40, 2000),
('uk_fine_tuned_small', 85.0, 90.0, 0.20, 800),
('uk_fine_tuned_medium', 90.0, 95.0, 0.30, 1200)
ON CONFLICT (model_name) DO NOTHING;

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metadata) VALUES
('overall_accuracy', 89.2, '{"baseline": 88.0}'::jsonb),
('uk_accuracy', 90.3, '{"baseline": 88.0}'::jsonb),
('ensemble_accuracy', 88.5, '{"baseline": 88.0}'::jsonb),
('patterns_learned', 5, '{"target": 100}'::jsonb)
ON CONFLICT DO NOTHING;

-- Insert initial activity
INSERT INTO activity_log (activity_type, message, metadata) VALUES
('system', '🚀 VoiceStand Learning System initialized with PostgreSQL storage', '{"version": "1.0.0"}'::jsonb);