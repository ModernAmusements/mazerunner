#!/usr/bin/env python3
"""
Behavioral Maze CAPTCHA System - Startup Validation & Health Check Script
Ensures all system components are operational before application launch
"""

import sys
import socket
import subprocess
import sqlite3
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import datetime

class SystemValidator:
    """Comprehensive system validation with zero-error policy"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/validation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_required_ports(self, ports: List[int]) -> bool:
        """Verify all required ports are available"""
        self.logger.info("🔍 Checking required ports...")
        
        for port in ports:
            self.total_checks += 1
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex(('127.0.0.1', port))
                    if result == 0:
                        self.errors.append(f"Port {port} is already in use")
                        self.logger.error(f"❌ Port {port} is occupied")
                    else:
                        self.success_count += 1
                        self.logger.info(f"✅ Port {port} is available")
            except Exception as e:
                self.errors.append(f"Error checking port {port}: {str(e)}")
                self.logger.error(f"❌ Failed to check port {port}: {e}")
        
        return len([e for e in self.errors if 'Port' in e]) == 0
    
    def check_python_dependencies(self) -> bool:
        """Verify all required Python packages are importable"""
        self.logger.info("🐍 Checking Python dependencies...")
        
        required_packages = [
            'numpy', 'cv2', 'flask', 'sqlite3', 
            'hashlib', 'json', 'math', 'random',
            'time', 'datetime', 'collections'
        ]
        
        for package in required_packages:
            self.total_checks += 1
            try:
                if package == 'cv2':
                    importlib.import_module('cv2')
                elif package == 'sqlite3':
                    importlib.import_module('sqlite3')
                else:
                    importlib.import_module(package)
                self.success_count += 1
                self.logger.info(f"✅ {package} imported successfully")
            except ImportError as e:
                self.errors.append(f"Missing package: {package} - {str(e)}")
                self.logger.error(f"❌ Failed to import {package}: {e}")
        
        return len([e for e in self.errors if 'Missing package' in e]) == 0
    
    def check_database_connection(self) -> bool:
        """Verify database connectivity and schema"""
        self.logger.info("💾 Checking database connection...")
        
        self.total_checks += 1
        try:
            # Check if database file exists or can be created
            db_path = 'maze_captcha.db'
            
            # Test database operations
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            if result and result[0] == 1:
                self.success_count += 1
                self.logger.info("✅ Database connection successful")
                
                # Check if tables exist, create if not
                self.initialize_database_schema(cursor)
                conn.commit()
                conn.close()
                return True
            else:
                self.errors.append("Database test query failed")
                self.logger.error("❌ Database test query failed")
                return False
                
        except sqlite3.Error as e:
            self.errors.append(f"Database connection failed: {str(e)}")
            self.logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def initialize_database_schema(self, cursor) -> None:
        """Initialize database schema with required tables"""
        schema_queries = [
            """
            CREATE TABLE IF NOT EXISTS captcha_sessions (
                id TEXT PRIMARY KEY,
                maze_data TEXT NOT NULL,
                solution_path TEXT NOT NULL,
                start_point TEXT NOT NULL,
                end_point TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_verified BOOLEAN DEFAULT FALSE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                coordinates TEXT NOT NULL,
                timestamp_data TEXT NOT NULL,
                velocity_data TEXT,
                solve_time REAL,
                is_human BOOLEAN,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES captcha_sessions (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                session_id TEXT,
                data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS bot_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                indicator_type TEXT NOT NULL,
                value REAL NOT NULL,
                threshold REAL,
                is_flagged BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES captcha_sessions (id)
            )
            """
        ]
        
        for query in schema_queries:
            try:
                cursor.execute(query)
            except sqlite3.Error as e:
                self.warnings.append(f"Schema initialization warning: {str(e)}")
                self.logger.warning(f"⚠️ Schema warning: {e}")
    
    def check_filesystem_permissions(self) -> bool:
        """Verify filesystem permissions for required directories"""
        self.logger.info("📁 Checking filesystem permissions...")
        
        required_dirs = ['logs', 'templates']
        required_files = ['requirements.txt']
        
        for directory in required_dirs:
            self.total_checks += 1
            try:
                Path(directory).mkdir(exist_ok=True)
                # Test write permissions
                test_file = Path(directory) / 'test_write.tmp'
                test_file.write_text('test')
                test_file.unlink()
                self.success_count += 1
                self.logger.info(f"✅ Directory {directory} is writable")
            except Exception as e:
                self.errors.append(f"Cannot write to directory {directory}: {str(e)}")
                self.logger.error(f"❌ Cannot write to {directory}: {e}")
        
        for file_path in required_files:
            self.total_checks += 1
            try:
                if Path(file_path).exists():
                    self.success_count += 1
                    self.logger.info(f"✅ File {file_path} exists")
                else:
                    self.errors.append(f"Required file {file_path} is missing")
                    self.logger.error(f"❌ Missing file: {file_path}")
            except Exception as e:
                self.errors.append(f"Error checking file {file_path}: {str(e)}")
                self.logger.error(f"❌ Error checking {file_path}: {e}")
        
        return len([e for e in self.errors if 'Directory' in e or 'Missing file' in e]) == 0
    
    def check_memory_requirements(self) -> bool:
        """Check system memory requirements"""
        self.total_checks += 1
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 0.5:  # 512MB minimum
                self.success_count += 1
                self.logger.info(f"✅ Available memory: {available_gb:.2f}GB")
                return True
            else:
                self.errors.append(f"Insufficient memory: {available_gb:.2f}GB available (min: 0.5GB)")
                self.logger.error(f"❌ Insufficient memory: {available_gb:.2f}GB")
                return False
                
        except ImportError:
            self.warnings.append("psutil not available - cannot check memory")
            self.logger.warning("⚠️ Memory check skipped (psutil not installed)")
            return True  # Don't fail if psutil is missing
    
    def run_validation(self) -> bool:
        """Execute all validation checks"""
        self.logger.info("🚀 Starting Behavioral Maze CAPTCHA System Validation")
        self.logger.info("=" * 60)
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Run all checks
        checks_passed = True
        
        checks_passed &= self.check_python_dependencies()
        checks_passed &= self.check_required_ports([8080])  # Default port
        checks_passed &= self.check_filesystem_permissions()
        checks_passed &= self.check_database_connection()
        checks_passed &= self.check_memory_requirements()
        
        # Generate report
        self.generate_validation_report()
        
        if not checks_passed or self.errors:
            self.logger.error("❌ VALIDATION FAILED - System not ready for launch")
            for error in self.errors:
                self.logger.error(f"   • {error}")
            return False
        else:
            self.logger.info("🎉 VALIDATION SUCCESSFUL - System ready for launch")
            if self.warnings:
                self.logger.warning("⚠️ Warnings detected:")
                for warning in self.warnings:
                    self.logger.warning(f"   • {warning}")
            return True
    
    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report"""
        self.logger.info("=" * 60)
        self.logger.info("📊 VALIDATION REPORT")
        self.logger.info(f"✅ Passed: {self.success_count}/{self.total_checks}")
        self.logger.info(f"❌ Errors: {len(self.errors)}")
        self.logger.info(f"⚠️ Warnings: {len(self.warnings)}")
        
        # Save report to file
        report = {
            "timestamp": str(datetime.datetime.now()),
            "success_rate": self.success_count / self.total_checks * 100,
            "passed_checks": self.success_count,
            "total_checks": self.total_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "system_ready": len(self.errors) == 0
        }
        
        try:
            with open('logs/validation_report.json', 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")

def main():
    """Main entry point for validation script"""
    validator = SystemValidator()
    
    if validator.run_validation():
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()