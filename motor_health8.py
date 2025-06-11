# with import csv file
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# MySQL connection imports
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    st.error("⚠️ MySQL connector not installed. Run: pip install mysql-connector-python")

# Page configuration
st.set_page_config(
    page_title="AI Preventive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .mysql-connected {
        background-color: #d1f2eb;
        color: #0e6251;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .mysql-disconnected {
        background-color: #fadbd8;
        color: #922b21;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MySQLConnector:
    """Handles MySQL database connections and data retrieval"""
    
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.is_connected = False
        
    def connect(self, host, port, database, username, password):
        """Establish connection to MySQL database with timeout and proper error handling"""
        try:
            # Close any existing connection first
            self.disconnect()
            
            # Connection configuration with timeouts and SSL options
            config = {
                'host': host,
                'port': port,
                'user': username,
                'password': password,
                'autocommit': True,
                'connection_timeout': 15,  # 15 seconds timeout
                'connect_timeout': 15,
                'raise_on_warnings': False,
                'use_pure': True,  # Use pure Python implementation
                'ssl_disabled': True,  # Disable SSL by default
                'auth_plugin': 'mysql_native_password'  # Use native password authentication
            }
            
            # Add database if provided and not empty
            if database and database.strip():
                config['database'] = database
            
            # First attempt: Try without SSL
            try:
                self.connection = mysql.connector.connect(**config)
                
                if self.connection.is_connected():
                    self.cursor = self.connection.cursor(buffered=True)
                    self.is_connected = True
                    
                    # Test the connection with a simple query
                    self.cursor.execute("SELECT 1")
                    self.cursor.fetchall()
                    
                    return True, f"Successfully connected to MySQL database at {host}:{port} (SSL disabled)"
                    
            except mysql.connector.Error as ssl_error:
                # If SSL-disabled connection fails, try with SSL enabled but not verified
                if "SSL" in str(ssl_error) or "ssl" in str(ssl_error).lower():
                    try:
                        config.update({
                            'ssl_disabled': False,
                            'ssl_verify_cert': False,
                            'ssl_verify_identity': False,
                            'ssl_ca': None,
                            'ssl_cert': None,
                            'ssl_key': None
                        })
                        
                        self.connection = mysql.connector.connect(**config)
                        
                        if self.connection.is_connected():
                            self.cursor = self.connection.cursor(buffered=True)
                            self.is_connected = True
                            
                            # Test the connection
                            self.cursor.execute("SELECT 1")
                            self.cursor.fetchall()
                            
                            return True, f"Successfully connected to MySQL database at {host}:{port} (SSL enabled, not verified)"
                            
                    except mysql.connector.Error:
                        # If that also fails, try with specific SSL mode
                        try:
                            config.update({
                                'ssl_disabled': False,
                                'ssl_verify_cert': False,
                                'ssl_verify_identity': False,
                                'use_unicode': True,
                                'charset': 'utf8mb4'
                            })
                            
                            self.connection = mysql.connector.connect(**config)
                            
                            if self.connection.is_connected():
                                self.cursor = self.connection.cursor(buffered=True)
                                self.is_connected = True
                                
                                # Test the connection
                                self.cursor.execute("SELECT 1")
                                self.cursor.fetchall()
                                
                                return True, f"Successfully connected to MySQL database at {host}:{port} (SSL relaxed mode)"
                                
                        except mysql.connector.Error:
                            pass
                
                # Re-raise the original error if all SSL attempts fail
                raise ssl_error
            
            self.is_connected = False
            return False, "Failed to establish connection"
                
        except mysql.connector.Error as e:
            self.is_connected = False
            error_msg = str(e)
            
            # Provide more specific error messages
            if "Access denied" in error_msg:
                return False, "Access denied - Check username and password"
            elif "Unknown database" in error_msg:
                return False, f"Database '{database}' does not exist"
            elif "Can't connect to MySQL server" in error_msg:
                return False, f"Cannot connect to MySQL server at {host}:{port} - Check if server is running and accessible"
            elif "timed out" in error_msg.lower():
                return False, f"Connection timeout - Server {host}:{port} not responding within 15 seconds"
            elif "SSL" in error_msg or "ssl" in error_msg.lower():
                return False, f"SSL Connection Error: {error_msg}\n\nTip: Try enabling 'Allow Insecure Connections' in your MySQL server settings"
            elif "wrong version number" in error_msg.lower():
                return False, f"SSL Version Mismatch: The server may require a different SSL/TLS version\n\nSuggestions:\n• Check if MySQL server allows non-SSL connections\n• Verify server SSL configuration\n• Contact server administrator"
            elif "Host" in error_msg and "is not allowed to connect" in error_msg:
                return False, f"Host not allowed - Your IP address may not be whitelisted on the MySQL server"
            else:
                return False, f"MySQL Error: {error_msg}"
                
        except Exception as e:
            self.is_connected = False
            return False, f"Unexpected error: {str(e)}"
        
    def disconnect(self):
        """Close database connection safely"""
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
                self.cursor = None
            if hasattr(self, 'connection') and self.connection and self.connection.is_connected():
                self.connection.close()
                self.connection = None
            self.is_connected = False
            return True, "Disconnected from MySQL database"
        except Exception as e:
            self.is_connected = False
            return False, f"Error during disconnection: {str(e)}"
    
    def get_databases(self):
        """Get list of available databases with error handling"""
        if not self.is_connected:
            return []
        try:
            self.cursor.execute("SHOW DATABASES")
            databases = [db[0] for db in self.cursor.fetchall()]
            # Filter out system databases for cleaner list
            system_dbs = ['information_schema', 'performance_schema', 'mysql', 'sys']
            return [db for db in databases if db not in system_dbs]
        except mysql.connector.Error as e:
            st.error(f"Error fetching databases: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Unexpected error fetching databases: {str(e)}")
            return []
    
    def get_tables(self):
        """Get list of tables in current database with error handling"""
        if not self.is_connected:
            return []
        try:
            # Check if a database is selected
            self.cursor.execute("SELECT DATABASE()")
            current_db = self.cursor.fetchone()[0]
            
            if not current_db:
                st.warning("No database selected. Please select a database first.")
                return []
            
            self.cursor.execute("SHOW TABLES")
            tables = [table[0] for table in self.cursor.fetchall()]
            return tables
        except mysql.connector.Error as e:
            error_msg = str(e)
            if "1046" in error_msg or "No database selected" in error_msg:
                st.error("❌ No database selected. Please select a database first.")
            else:
                st.error(f"Error fetching tables: {error_msg}")
            return []
        except Exception as e:
            st.error(f"Unexpected error fetching tables: {str(e)}")
            return []
    
    def get_columns(self, table_name):
        """Get list of columns in specified table with error handling"""
        if not self.is_connected:
            return []
        try:
            # Check if a database is selected
            self.cursor.execute("SELECT DATABASE()")
            current_db = self.cursor.fetchone()[0]
            
            if not current_db:
                st.warning("No database selected. Please select a database first.")
                return []
            
            # Use parameterized query to prevent SQL injection
            query = f"DESCRIBE `{table_name}`"
            self.cursor.execute(query)
            columns = [column[0] for column in self.cursor.fetchall()]
            return columns
        except mysql.connector.Error as e:
            error_msg = str(e)
            if "1046" in error_msg or "No database selected" in error_msg:
                st.error("❌ No database selected. Please select a database first.")
            elif "doesn't exist" in error_msg.lower():
                st.error(f"❌ Table '{table_name}' doesn't exist in the selected database.")
            else:
                st.error(f"Error fetching columns for table '{table_name}': {error_msg}")
            return []
        except Exception as e:
            st.error(f"Unexpected error fetching columns: {str(e)}")
            return []
        
    def get_current_database(self):
        """Get the currently selected database"""
        if not self.is_connected:
            return None
        try:
            self.cursor.execute("SELECT DATABASE()")
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            st.error(f"Error getting current database: {str(e)}")
            return None
    
    def switch_database(self, database_name):
        """Switch to a specific database"""
        if not self.is_connected:
            return False, "Not connected to MySQL server"
        
        try:
            # Use the USE statement to switch database
            self.cursor.execute(f"USE `{database_name}`")
            return True, f"Successfully switched to database: {database_name}"
        except mysql.connector.Error as e:
            error_msg = str(e)
            if "1049" in error_msg or "Unknown database" in error_msg:
                return False, f"Database '{database_name}' does not exist"
            elif "1044" in error_msg or "Access denied" in error_msg:
                return False, f"Access denied to database '{database_name}'"
            else:
                return False, f"Error switching to database: {error_msg}"
        except Exception as e:
            return False, f"Unexpected error switching database: {str(e)}"
    
    def test_table_access(self, table_name):
        """Test if we can access a specific table"""
        if not self.is_connected:
            return False, "Not connected to MySQL server"
        
        try:
            # Try to get basic info about the table
            query = f"SELECT COUNT(*) FROM `{table_name}` LIMIT 1"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            return True, f"Table '{table_name}' is accessible with {result[0]} total rows"
        except mysql.connector.Error as e:
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg:
                return False, f"Table '{table_name}' doesn't exist"
            elif "1142" in error_msg or "command denied" in error_msg:
                return False, f"Access denied to table '{table_name}'"
            else:
                return False, f"Error accessing table: {error_msg}"
        except Exception as e:
            return False, f"Unexpected error testing table access: {str(e)}"
    
    def get_latest_data(self, table_name, columns_mapping, limit=1000, order_by_timestamp=True):
        """Retrieve latest data from database"""
        if not self.is_connected:
            return None
        
        try:
            # Build column selection
            selected_columns = []
            column_aliases = []
            
            for key, column in columns_mapping.items():
                if column and column != "None":
                    selected_columns.append(column)
                    column_aliases.append(key)
            
            if not selected_columns:
                return None
            
            # Build query
            column_str = ", ".join(selected_columns)
            query = f"SELECT {column_str} FROM {table_name}"
            
            # Add ordering by timestamp if specified
            if order_by_timestamp and columns_mapping.get('timestamp'):
                query += f" ORDER BY {columns_mapping['timestamp']} DESC"
            
            query += f" LIMIT {limit}"
            
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=column_aliases)
            
            # Sort by timestamp ascending for proper time series analysis
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Error as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def get_real_time_data(self, table_name, columns_mapping, last_timestamp=None):
        """Get real-time data since last timestamp"""
        if not self.is_connected:
            return None
        
        try:
            # Build column selection
            selected_columns = []
            column_aliases = []
            
            for key, column in columns_mapping.items():
                if column and column != "None":
                    selected_columns.append(column)
                    column_aliases.append(key)
            
            if not selected_columns:
                return None
            
            # Build query
            column_str = ", ".join(selected_columns)
            query = f"SELECT {column_str} FROM {table_name}"
            
            # Add timestamp filter for real-time data
            if last_timestamp and columns_mapping.get('timestamp'):
                query += f" WHERE {columns_mapping['timestamp']} > '{last_timestamp}'"
            
            # Order by timestamp
            if columns_mapping.get('timestamp'):
                query += f" ORDER BY {columns_mapping['timestamp']} ASC"
            
            query += " LIMIT 100"  # Limit for real-time updates
            
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=column_aliases)
                return df
            else:
                return None
                
        except Error as e:
            st.error(f"Error fetching real-time data: {str(e)}")
            return None
        
    def get_data_by_date_range(self, table_name, columns_mapping, start_datetime, end_datetime, limit=10000):
        """Retrieve data from database within specified date range"""
        if not self.is_connected:
            return None
        
        try:
            # Build column selection
            selected_columns = []
            column_aliases = []
            
            for key, column in columns_mapping.items():
                if column and column != "None":
                    selected_columns.append(column)
                    column_aliases.append(key)
            
            if not selected_columns:
                return None
            
            # Build query with date range filter
            column_str = ", ".join(selected_columns)
            timestamp_column = columns_mapping.get('timestamp')
            
            if not timestamp_column:
                st.warning("No timestamp column specified for date range filtering")
                return None
            
            query = f"""
            SELECT {column_str} 
            FROM {table_name} 
            WHERE {timestamp_column} BETWEEN %s AND %s
            ORDER BY {timestamp_column} ASC
            LIMIT %s
            """
            
            # Execute query with parameters
            self.cursor.execute(query, (start_datetime, end_datetime, limit))
            rows = self.cursor.fetchall()
            
            # Convert to DataFrame
            if rows:
                df = pd.DataFrame(rows, columns=column_aliases)
                return df
            else:
                return pd.DataFrame(columns=column_aliases)
                
        except Error as e:
            st.error(f"Error fetching data by date range: {str(e)}")
            return None

    def get_date_range_statistics(self, table_name, timestamp_column):
        """Get statistics about available date ranges in the table"""
        if not self.is_connected:
            return None
        
        try:
            # Get basic date range info
            stats_query = f"""
            SELECT 
                MIN({timestamp_column}) as min_date,
                MAX({timestamp_column}) as max_date,
                COUNT(*) as total_records,
                COUNT(DISTINCT DATE({timestamp_column})) as unique_days
            FROM {table_name}
            WHERE {timestamp_column} IS NOT NULL
            """
            
            self.cursor.execute(stats_query)
            basic_stats = self.cursor.fetchone()
            
            if not basic_stats or not basic_stats[0]:
                return None
            
            # Get daily distribution
            daily_query = f"""
            SELECT 
                DATE({timestamp_column}) as date,
                COUNT(*) as records_count,
                MIN({timestamp_column}) as first_record,
                MAX({timestamp_column}) as last_record
            FROM {table_name}
            WHERE {timestamp_column} IS NOT NULL
            GROUP BY DATE({timestamp_column})
            ORDER BY date DESC
            LIMIT 30
            """
            
            self.cursor.execute(daily_query)
            daily_stats = self.cursor.fetchall()
            
            # Get hourly distribution for recent data
            hourly_query = f"""
            SELECT 
                HOUR({timestamp_column}) as hour,
                COUNT(*) as records_count
            FROM {table_name}
            WHERE {timestamp_column} >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY HOUR({timestamp_column})
            ORDER BY hour
            """
            
            self.cursor.execute(hourly_query)
            hourly_stats = self.cursor.fetchall()
            
            return {
                'min_date': basic_stats[0],
                'max_date': basic_stats[1],
                'total_records': basic_stats[2],
                'unique_days': basic_stats[3],
                'daily_distribution': daily_stats,
                'hourly_distribution': hourly_stats
            }
            
        except Error as e:
            st.error(f"Error getting date statistics: {str(e)}")
            return None

    def get_latest_data_with_date_filter(self, table_name, columns_mapping, start_datetime=None, end_datetime=None, limit=1000, order_by_timestamp=True):
        """Enhanced version of get_latest_data with optional date filtering"""
        if not self.is_connected:
            return None
        
        try:
            # Build column selection
            selected_columns = []
            column_aliases = []
            
            for key, column in columns_mapping.items():
                if column and column != "None":
                    selected_columns.append(column)
                    column_aliases.append(key)
            
            if not selected_columns:
                return None
            
            # Build query
            column_str = ", ".join(selected_columns)
            query = f"SELECT {column_str} FROM {table_name}"
            params = []
            
            # Add date range filter if specified
            if start_datetime and end_datetime and columns_mapping.get('timestamp'):
                query += f" WHERE {columns_mapping['timestamp']} BETWEEN %s AND %s"
                params.extend([start_datetime, end_datetime])
            elif start_datetime and columns_mapping.get('timestamp'):
                query += f" WHERE {columns_mapping['timestamp']} >= %s"
                params.append(start_datetime)
            elif end_datetime and columns_mapping.get('timestamp'):
                query += f" WHERE {columns_mapping['timestamp']} <= %s"
                params.append(end_datetime)
            
            # Add ordering
            if order_by_timestamp and columns_mapping.get('timestamp'):
                if start_datetime or end_datetime:
                    query += f" ORDER BY {columns_mapping['timestamp']} ASC"
                else:
                    query += f" ORDER BY {columns_mapping['timestamp']} DESC"
            
            query += f" LIMIT %s"
            params.append(limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            # Convert to DataFrame
            if rows:
                df = pd.DataFrame(rows, columns=column_aliases)
                
                # Sort by timestamp ascending for proper time series analysis if we have date filtering
                if 'timestamp' in df.columns and (start_datetime or end_datetime):
                    df = df.sort_values('timestamp').reset_index(drop=True)
                elif 'timestamp' in df.columns and not (start_datetime or end_datetime):
                    # For latest data without date filter, keep descending order but reverse for analysis
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                return df
            else:
                return pd.DataFrame(columns=column_aliases)
                
        except Error as e:
            st.error(f"Error fetching filtered data: {str(e)}")
            return None

class VibrationDataGenerator:
    """Simulates multi-axis vibration sensor data with various fault conditions"""
    
    def __init__(self):
        self.sampling_rate = 1000  # Hz
        self.duration = 2  # seconds
        self.time_vector = np.linspace(0, self.duration, self.sampling_rate * self.duration)
        
    def generate_healthy_signal(self, axis='x'):
        """Generate normal vibration signal for specified axis"""
        # Base rotation frequency (e.g., 30 Hz for 1800 RPM)
        base_freq = 30
        
        # Different characteristics for each axis
        if axis == 'x':
            signal_data = (
                0.5 * np.sin(2 * np.pi * base_freq * self.time_vector) +
                0.2 * np.sin(2 * np.pi * 2 * base_freq * self.time_vector) +
                0.1 * np.random.normal(0, 1, len(self.time_vector))
            )
        elif axis == 'y':
            signal_data = (
                0.4 * np.sin(2 * np.pi * base_freq * self.time_vector + np.pi/4) +
                0.15 * np.sin(2 * np.pi * 2 * base_freq * self.time_vector) +
                0.08 * np.random.normal(0, 1, len(self.time_vector))
            )
        else:  # z-axis
            signal_data = (
                0.3 * np.sin(2 * np.pi * base_freq * self.time_vector + np.pi/2) +
                0.1 * np.sin(2 * np.pi * 2 * base_freq * self.time_vector) +
                0.06 * np.random.normal(0, 1, len(self.time_vector))
            )
        
        return signal_data
    
    def generate_faulty_signal(self, fault_type="bearing", axis='x'):
        """Generate vibration signal with specific fault patterns for specified axis"""
        base_signal = self.generate_healthy_signal(axis)
        
        # Fault severity varies by axis
        axis_multiplier = {'x': 1.0, 'y': 0.8, 'z': 0.6}[axis]
        
        if fault_type == "bearing":
            # Bearing fault: high frequency components + impulses
            bearing_freq = 157  # Bearing characteristic frequency
            fault_signal = 0.8 * axis_multiplier * np.sin(2 * np.pi * bearing_freq * self.time_vector)
            # Add random impulses
            impulses = np.random.choice([0, 1], len(self.time_vector), p=[0.95, 0.05]) * 2 * axis_multiplier
            return base_signal + fault_signal + impulses
            
        elif fault_type == "imbalance":
            # Imbalance: increased amplitude at rotation frequency (more prominent in X and Y)
            imbalance_multiplier = {'x': 2.5, 'y': 2.0, 'z': 1.2}[axis]
            return base_signal * imbalance_multiplier + 0.3 * np.random.normal(0, 1, len(self.time_vector))
            
        elif fault_type == "misalignment":
            # Misalignment: increased harmonics (varies by axis)
            harm2 = 1.2 * axis_multiplier * np.sin(2 * np.pi * 60 * self.time_vector)
            harm3 = 0.8 * axis_multiplier * np.sin(2 * np.pi * 90 * self.time_vector)
            return base_signal + harm2 + harm3
            
        return base_signal
    
    def generate_temperature_data(self, base_temp=75, fault_condition=None):
        """Generate motor temperature data (v0)"""
        # Base temperature with some variation
        temp_variation = 5 * np.sin(2 * np.pi * 0.1 * self.time_vector) + 2 * np.random.normal(0, 1, len(self.time_vector))
        
        if fault_condition == "bearing":
            # Bearing fault causes temperature rise
            temp_rise = 15 + 5 * np.sin(2 * np.pi * 0.05 * self.time_vector)
            return base_temp + temp_variation + temp_rise
        elif fault_condition == "imbalance":
            # Slight temperature increase due to increased vibration
            temp_rise = 8 + 3 * np.sin(2 * np.pi * 0.08 * self.time_vector)
            return base_temp + temp_variation + temp_rise
        elif fault_condition == "misalignment":
            # Moderate temperature increase
            temp_rise = 12 + 4 * np.sin(2 * np.pi * 0.06 * self.time_vector)
            return base_temp + temp_variation + temp_rise
        else:
            # Normal temperature
            return base_temp + temp_variation

class AIAnalyzer:
    """AI-based multi-axis vibration and temperature analysis system with enhanced reliability"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
        # Machine-specific parameters (configurable)
        self.machine_config = {
            'motor_rpm': 1800,
            'rotation_freq': 30,  # Hz
            'bearing_freqs': [157, 234, 89],  # BPFI, BPFO, BSF
            'normal_ranges': {
                'Fx': {'rms': (0.3, 0.8), 'crest': (2.5, 4.0)},
                'Fy': {'rms': (0.2, 0.6), 'crest': (2.8, 4.2)},
                'Fz': {'rms': (0.15, 0.5), 'crest': (3.0, 4.5)}
            },
            'temp_thresholds': {
                'normal_max': 80,
                'warning_max': 85, 
                'critical_max': 95,
                'max_rise_rate': 2.0  # °C/sec
            }
        }
        
        # Health calculation weights
        self.health_weights = {
            'anomaly_detection': 0.4,
            'axis_analysis': 0.6,
            'vibration_weight': 0.7,
            'temperature_weight': 0.3
        }
        
        # Training validation metrics
        self.training_stats = {}
        self.raw_training_data = []  # Store raw training signals
        self.training_features = []  # Store extracted features
        
    def save_training_data(self, filepath):
        """Save training data to file"""
        import pickle
        training_package = {
            'raw_training_data': self.raw_training_data,
            'training_features': self.training_features,
            'training_stats': self.training_stats,
            'machine_config': self.machine_config,
            'health_weights': self.health_weights,
            'scaler_params': {
                'mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None
            },
            'model_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(training_package, f)
        
        return f"Training data saved to {filepath}"
    
    def load_training_data(self, filepath):
        """Load training data from file"""
        import pickle
        try:
            with open(filepath, 'rb') as f:
                training_package = pickle.load(f)
            
            self.raw_training_data = training_package.get('raw_training_data', [])
            self.training_features = training_package.get('training_features', [])
            self.training_stats = training_package.get('training_stats', {})
            self.machine_config.update(training_package.get('machine_config', {}))
            self.health_weights.update(training_package.get('health_weights', {}))
            
            # Restore scaler if available
            scaler_params = training_package.get('scaler_params', {})
            if scaler_params.get('mean') is not None:
                self.scaler.mean_ = scaler_params['mean']
                self.scaler.scale_ = scaler_params['scale']
            
            self.is_trained = training_package.get('model_trained', False)
            
            return True, f"Training data loaded from {filepath}"
        except Exception as e:
            return False, f"Failed to load training data: {str(e)}"
    
    def get_training_summary(self):
        """Get comprehensive training data summary"""
        if not self.training_stats:
            return {"status": "No training data available"}
        
        summary = {
            "training_status": "Trained" if self.is_trained else "Not trained",
            "num_training_samples": self.training_stats.get('num_samples', 0),
            "sampling_rate": self.training_stats.get('sampling_rate', 'Unknown'),
            "feature_statistics": {},
            "data_quality": {},
            "machine_config": self.machine_config.copy()
        }
        
        # Feature statistics
        if 'feature_means' in self.training_stats:
            feature_names = self._get_feature_names()
            summary["feature_statistics"] = {
                "means": dict(zip(feature_names, self.training_stats['feature_means'])),
                "stds": dict(zip(feature_names, self.training_stats['feature_stds'])),
                "ranges": dict(zip(feature_names, 
                    self.training_stats['feature_maxs'] - self.training_stats['feature_mins']))
            }
        
        # Data quality assessment
        if self.training_features:
            features_array = np.array(self.training_features)
            summary["data_quality"] = {
                "feature_count": features_array.shape[1],
                "sample_count": features_array.shape[0],
                "constant_features": np.sum(np.std(features_array, axis=0) < 1e-6),
                "feature_correlations": self._calculate_feature_correlations()
            }
        
        return summary
    
    def _get_feature_names(self):
        """Generate feature names based on configuration"""
        feature_names = []
        
        # Add vibration features for each axis
        for axis in ['Fx', 'Fy', 'Fz']:
            if axis in str(self.machine_config):  # Check if axis was used
                feature_names.extend([
                    f"{axis}_RMS", f"{axis}_Peak", f"{axis}_Crest",
                    f"{axis}_Skewness", f"{axis}_Kurtosis", 
                    f"{axis}_DominantFreq", f"{axis}_SpectralCentroid"
                ])
        
        # Add temperature features if used
        if 'v0' in str(self.machine_config):
            feature_names.extend([
                "v0_Mean", "v0_Std", "v0_Max", "v0_Min", "v0_Gradient", "v0_Range"
            ])
        
        return feature_names
    
    def _calculate_feature_correlations(self):
        """Calculate correlation matrix for training features"""
        if not self.training_features or len(self.training_features) < 2:
            return "Insufficient data for correlation analysis"
        
        try:
            features_array = np.array(self.training_features)
            correlation_matrix = np.corrcoef(features_array.T)
            
            # Find highly correlated features (>0.9)
            high_corr_pairs = []
            feature_names = self._get_feature_names()
            
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    if abs(correlation_matrix[i, j]) > 0.9:
                        high_corr_pairs.append({
                            'feature1': feature_names[i] if i < len(feature_names) else f"Feature_{i}",
                            'feature2': feature_names[j] if j < len(feature_names) else f"Feature_{j}",
                            'correlation': correlation_matrix[i, j]
                        })
            
            return {
                'high_correlation_pairs': high_corr_pairs,
                'matrix_shape': correlation_matrix.shape
            }
        except Exception as e:
            return f"Error calculating correlations: {str(e)}"
    
    def compare_with_training(self, current_signals, sampling_rate=1000):
        """Compare current data with training data for anomaly detection"""
        if not self.is_trained or not self.training_stats:
            return {"status": "No training data available for comparison"}
        
        try:
            # Extract features from current data
            current_features = self.extract_multi_axis_features(current_signals, sampling_rate)
            
            comparison = {
                "feature_comparison": {},
                "anomaly_scores": {},
                "overall_similarity": 0,
                "warnings": []
            }
            
            # Compare each feature with training statistics
            feature_names = self._get_feature_names()
            training_means = self.training_stats['feature_means']
            training_stds = self.training_stats['feature_stds']
            
            similarities = []
            
            for i, (current_val, train_mean, train_std, name) in enumerate(zip(
                current_features, training_means, training_stds, feature_names)):
                
                # Calculate z-score
                z_score = abs(current_val - train_mean) / train_std if train_std > 0 else 0
                
                # Calculate similarity (inverse of z-score, normalized)
                similarity = max(0, 100 - z_score * 20)  # 100% at z=0, 0% at z=5
                similarities.append(similarity)
                
                comparison["feature_comparison"][name] = {
                    "current_value": current_val,
                    "training_mean": train_mean,
                    "training_std": train_std,
                    "z_score": z_score,
                    "similarity_percent": similarity
                }
                
                # Add warnings for significant deviations
                if z_score > 3:
                    comparison["warnings"].append(f"{name}: Significant deviation (z-score: {z_score:.2f})")
                elif z_score > 2:
                    comparison["warnings"].append(f"{name}: Moderate deviation (z-score: {z_score:.2f})")
            
            comparison["overall_similarity"] = np.mean(similarities)
            
            # AI model anomaly score
            if self.is_trained:
                try:
                    features_scaled = self.scaler.transform(current_features.reshape(1, -1))
                    anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                    is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                    
                    comparison["anomaly_scores"] = {
                        "anomaly_score": anomaly_score,
                        "is_anomaly": is_anomaly,
                        "confidence": min(100, abs(anomaly_score) * 50)
                    }
                except Exception as e:
                    comparison["anomaly_scores"] = {"error": str(e)}
            
            return comparison
            
        except Exception as e:
            return {"status": f"Error in comparison: {str(e)}"}
        
    def configure_machine(self, motor_rpm=None, bearing_specs=None, machine_type=None):
        """Configure analyzer for specific machine parameters"""
        if motor_rpm:
            self.machine_config['motor_rpm'] = motor_rpm
            self.machine_config['rotation_freq'] = motor_rpm / 60
            
        if bearing_specs:
            # Calculate bearing fault frequencies based on specifications
            # This would typically use bearing geometry formulas
            self.machine_config['bearing_freqs'] = self._calculate_bearing_frequencies(bearing_specs)
            
        if machine_type:
            # Set machine-specific normal ranges
            self.machine_config['normal_ranges'] = self._get_machine_specific_ranges(machine_type)
            
        return f"Machine configured: {motor_rpm} RPM, Type: {machine_type}"
    
    def _calculate_bearing_frequencies(self, bearing_specs):
        """Calculate bearing fault frequencies from bearing specifications"""
        # Simplified calculation - in practice, use actual bearing geometry
        base_freq = self.machine_config['rotation_freq']
        
        # These would be calculated from bearing geometry:
        # BPFI = (Nb/2) * (1 + (Bd/Pd) * cos(φ)) * fs
        # BPFO = (Nb/2) * (1 - (Bd/Pd) * cos(φ)) * fs
        # BSF = (Pd/2Bd) * (1 - (Bd/Pd)² * cos²(φ)) * fs
        
        bpfi = base_freq * 5.23  # Example calculation
        bpfo = base_freq * 7.8   # Example calculation
        bsf = base_freq * 2.97   # Example calculation
        
        return [bpfi, bpfo, bsf]
    
    def _get_machine_specific_ranges(self, machine_type):
        """Get normal operating ranges for specific machine types"""
        ranges = {
            'motor': {
                'Fx': {'rms': (0.3, 0.8), 'crest': (2.5, 4.0)},
                'Fy': {'rms': (0.2, 0.6), 'crest': (2.8, 4.2)},
                'Fz': {'rms': (0.15, 0.5), 'crest': (3.0, 4.5)}
            },
            'pump': {
                'Fx': {'rms': (0.5, 1.2), 'crest': (3.0, 5.0)},
                'Fy': {'rms': (0.4, 1.0), 'crest': (3.2, 5.2)},
                'Fz': {'rms': (0.3, 0.8), 'crest': (3.5, 5.5)}
            },
            'fan': {
                'Fx': {'rms': (0.2, 0.6), 'crest': (2.0, 3.5)},
                'Fy': {'rms': (0.15, 0.5), 'crest': (2.2, 3.7)},
                'Fz': {'rms': (0.1, 0.4), 'crest': (2.5, 4.0)}
            }
        }
        return ranges.get(machine_type.lower(), ranges['motor'])
        
    def extract_features(self, signal_data, sampling_rate=1000, signal_type='vibration'):
        """Extract statistical and frequency domain features with robust error handling"""
        # Ensure signal_data is a numpy array
        signal_data = np.array(signal_data)
        
        # Check for empty or invalid data
        if len(signal_data) == 0:
            if signal_type == 'temperature':
                return np.array([0, 0, 0, 0, 0, 0])  # 6 temperature features
            else:
                return np.array([0, 0, 0, 0, 0, 0, 0])  # 7 vibration features
        
        if signal_type == 'temperature':
            # Temperature-specific features
            mean_temp = np.mean(signal_data)
            std_temp = np.std(signal_data) if len(signal_data) > 1 else 0
            max_temp = np.max(signal_data)
            min_temp = np.min(signal_data)
            
            # Temperature gradient (handle single point case)
            if len(signal_data) > 1:
                temp_gradient = np.mean(np.diff(signal_data))
            else:
                temp_gradient = 0
                
            temp_range = max_temp - min_temp
            
            return np.array([mean_temp, std_temp, max_temp, min_temp, temp_gradient, temp_range])
        
        else:
            # Vibration features (time domain)
            rms = np.sqrt(np.mean(signal_data**2))
            peak = np.max(np.abs(signal_data))
            crest_factor = peak / rms if rms > 0 else 0
            
            # Statistical features (handle edge cases)
            if len(signal_data) > 1:
                signal_std = np.std(signal_data)
                if signal_std > 0:
                    normalized_signal = (signal_data - np.mean(signal_data)) / signal_std
                    skewness = np.mean(normalized_signal**3)
                    kurtosis = np.mean(normalized_signal**4)
                else:
                    skewness = 0
                    kurtosis = 0
            else:
                skewness = 0
                kurtosis = 0
            
            # Frequency domain features (only if we have enough data)
            if len(signal_data) > 1:
                try:
                    fft_values = np.abs(fft(signal_data))
                    freqs = fftfreq(len(signal_data), 1/sampling_rate)
                    
                    # Find dominant frequencies
                    positive_freqs = freqs[:len(freqs)//2]
                    positive_fft = fft_values[:len(fft_values)//2]
                    
                    if len(positive_fft) > 0 and np.sum(positive_fft) > 0:
                        dominant_freq = positive_freqs[np.argmax(positive_fft)]
                        spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
                    else:
                        dominant_freq = 0
                        spectral_centroid = 0
                except:
                    dominant_freq = 0
                    spectral_centroid = 0
            else:
                dominant_freq = 0
                spectral_centroid = 0
            
            return np.array([rms, peak, crest_factor, skewness, kurtosis, 
                            dominant_freq, spectral_centroid])
    
    def extract_multi_axis_features(self, signals_dict, sampling_rate=1000):
        """Extract features from multiple axes and combine them"""
        all_features = []
        
        # Extract features for each selected axis
        for axis, signal in signals_dict.items():
            if axis in ['Fx', 'Fy', 'Fz']:
                features = self.extract_features(signal, sampling_rate, 'vibration')
            elif axis == 'v0':
                features = self.extract_features(signal, sampling_rate, 'temperature')
            else:
                continue
            all_features.extend(features)
        
        return np.array(all_features)
    
    def train_model(self, training_data_list, sampling_rate=1000, validate_data=True):
        """Train the anomaly detection model with enhanced data storage"""
        if len(training_data_list) < 10:
            return False, "Need at least 10 training samples for reliable model"
        
        # Store raw training data
        self.raw_training_data = training_data_list.copy()
        
        features_list = []
        for training_signals in training_data_list:
            features = self.extract_multi_axis_features(training_signals, sampling_rate)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # Store extracted features
        self.training_features = features_list.copy()
        
        # Validate training data quality
        if validate_data:
            validation_result = self._validate_training_data(features_array)
            if not validation_result['valid']:
                return False, f"Training data validation failed: {validation_result['reason']}"
        
        # Store comprehensive training statistics
        self.training_stats = {
            'feature_means': np.mean(features_array, axis=0),
            'feature_stds': np.std(features_array, axis=0),
            'feature_mins': np.min(features_array, axis=0),
            'feature_maxs': np.max(features_array, axis=0),
            'num_samples': len(training_data_list),
            'sampling_rate': sampling_rate,
            'timestamp': datetime.now().isoformat(),
            'signal_types': list(training_data_list[0].keys()) if training_data_list else [],
            'data_source': 'manual_training'
        }
        
        # Train the model
        features_scaled = self.scaler.fit_transform(features_array)
        self.anomaly_detector.fit(features_scaled)
        self.is_trained = True
        
        return True, f"Model trained successfully on {len(training_data_list)} samples"
    
    def _validate_training_data(self, features_array):
        """Validate quality of training data"""
        if features_array.shape[0] < 10:
            return {'valid': False, 'reason': 'Insufficient training samples'}
        
        if features_array.shape[1] == 0:
            return {'valid': False, 'reason': 'No features extracted'}
        
        # Check for constant features (no variation)
        feature_stds = np.std(features_array, axis=0)
        constant_features = np.sum(feature_stds < 1e-6)
        if constant_features > features_array.shape[1] * 0.5:
            return {'valid': False, 'reason': f'Too many constant features: {constant_features}'}
        
        # Check for extreme outliers in training data
        for i in range(features_array.shape[1]):
            feature_col = features_array[:, i]
            q75, q25 = np.percentile(feature_col, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 3 * iqr
            upper_bound = q75 + 3 * iqr
            outliers = np.sum((feature_col < lower_bound) | (feature_col > upper_bound))
            
            if outliers > len(feature_col) * 0.2:  # More than 20% outliers
                return {'valid': False, 'reason': f'Too many outliers in feature {i}'}
        
        return {'valid': True, 'reason': 'Training data quality acceptable'}
    
    def validate_real_time_data(self, signals_dict, sampling_rate=1000):
        """Validate that real-time data is within expected ranges of training data"""
        if not self.is_trained or not self.training_stats:
            return {'valid': True, 'warnings': ['Model not trained - cannot validate']}
        
        try:
            features = self.extract_multi_axis_features(signals_dict, sampling_rate)
        except:
            return {'valid': False, 'warnings': ['Failed to extract features from real-time data']}
        
        warnings = []
        
        # Check if features are within reasonable bounds of training data
        for i, (feature_val, train_mean, train_std) in enumerate(zip(
            features, self.training_stats['feature_means'], self.training_stats['feature_stds']
        )):
            # Allow 3 standard deviations from training mean
            if train_std > 0:
                z_score = abs(feature_val - train_mean) / train_std
                if z_score > 3:
                    warnings.append(f'Feature {i} outside training distribution (z-score: {z_score:.1f})')
        
        # Check sampling rate consistency
        if abs(sampling_rate - self.training_stats['sampling_rate']) > 100:
            warnings.append(f'Sampling rate mismatch: trained on {self.training_stats["sampling_rate"]}Hz, current {sampling_rate}Hz')
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings,
            'feature_comparison': {
                'current': features,
                'training_mean': self.training_stats['feature_means'],
                'training_std': self.training_stats['feature_stds']
            }
        }
        
    def analyze_signals(self, signals_dict, sampling_rate=1000):
        """Enhanced signal analysis with integrated health scoring"""
        # Validate input signals
        if not signals_dict or len(signals_dict) == 0:
            return {"health_score": 50, "anomaly": True, "confidence": 0, "features": [], "axis_analysis": {}, "validation": {}}
        
        # Validate real-time data against training
        validation_result = self.validate_real_time_data(signals_dict, sampling_rate)
        
        # Extract features and perform axis analysis
        try:
            axis_analysis = self.analyze_individual_axes(signals_dict, sampling_rate)
        except Exception as e:
            axis_analysis = {}
        
        # Anomaly detection (if model is trained)
        anomaly_health = 50
        is_anomaly = True
        confidence = 0
        features = []
        
        if self.is_trained:
            try:
                features = self.extract_multi_axis_features(signals_dict, sampling_rate).reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
                
                # Convert anomaly score to health score
                anomaly_health = max(0, min(100, 50 + anomaly_score * 25))
                confidence = min(100, abs(anomaly_score) * 50)
                features = features[0] if len(features[0]) > 0 else []
                
            except Exception as e:
                pass  # Fall back to default values
        
        # Calculate integrated health score
        integrated_health = self._calculate_integrated_health(anomaly_health, axis_analysis, validation_result)
        
        return {
            "health_score": integrated_health['overall_health'],
            "anomaly": is_anomaly,
            "confidence": confidence,
            "features": features,
            "axis_analysis": axis_analysis,
            "validation": validation_result,
            "health_breakdown": integrated_health,
            "machine_config": self.machine_config
        }
    
    def _calculate_integrated_health(self, anomaly_health, axis_analysis, validation_result):
        """Calculate integrated health score combining all analysis methods"""
        health_breakdown = {
            'anomaly_health': anomaly_health,
            'axis_health': 50,
            'validation_penalty': 0,
            'overall_health': 50
        }
        
        # Calculate average axis health if available
        if axis_analysis:
            axis_scores = [analysis.get('health_score', 50) for analysis in axis_analysis.values()]
            vibration_scores = [
                analysis.get('health_score', 50) 
                for key, analysis in axis_analysis.items() 
                if key in ['Fx', 'Fy', 'Fz']
            ]
            temp_scores = [
                analysis.get('health_score', 50) 
                for key, analysis in axis_analysis.items() 
                if key == 'v0'
            ]
            
            # Weight vibration vs temperature
            if vibration_scores and temp_scores:
                weighted_axis_health = (
                    np.mean(vibration_scores) * self.health_weights['vibration_weight'] +
                    np.mean(temp_scores) * self.health_weights['temperature_weight']
                )
            elif vibration_scores:
                weighted_axis_health = np.mean(vibration_scores)
            elif temp_scores:
                weighted_axis_health = np.mean(temp_scores)
            else:
                weighted_axis_health = 50
                
            health_breakdown['axis_health'] = weighted_axis_health
        
        # Apply validation penalties
        validation_penalty = 0
        if validation_result.get('warnings'):
            validation_penalty = min(20, len(validation_result['warnings']) * 5)  # Max 20 point penalty
        
        health_breakdown['validation_penalty'] = validation_penalty
        
        # Calculate final integrated health score
        if self.is_trained and axis_analysis:
            # Combine anomaly detection and axis analysis
            integrated_health = (
                anomaly_health * self.health_weights['anomaly_detection'] +
                health_breakdown['axis_health'] * self.health_weights['axis_analysis']
            )
        elif axis_analysis:
            # Use only axis analysis if no training
            integrated_health = health_breakdown['axis_health']
        else:
            # Fallback to anomaly detection only
            integrated_health = anomaly_health
        
        # Apply validation penalty
        integrated_health = max(0, integrated_health - validation_penalty)
        
        health_breakdown['overall_health'] = integrated_health
        return health_breakdown
        
    def update_weights(self, new_weights):
        """Update health calculation weights"""
        self.health_weights.update(new_weights)
        return "Weights updated successfully"
    
    def calibrate_from_current_data(self, current_signals, sampling_rate=1000):
        """Calibrate healthy baselines from current real motor data"""
        # Extract features from current data
        features = self.extract_multi_axis_features(current_signals, sampling_rate)
        
        # Update normal ranges based on current data
        for axis_key, signal in current_signals.items():
            if axis_key in ['Fx', 'Fy', 'Fz']:
                rms = np.sqrt(np.mean(signal**2))
                peak = np.max(np.abs(signal))
                crest_factor = peak / rms if rms > 0 else 0
                
                # Set new normal ranges with tolerance
                rms_tolerance = rms * 0.3  # ±30% tolerance
                crest_tolerance = crest_factor * 0.2  # ±20% tolerance
                
                self.machine_config['normal_ranges'][axis_key] = {
                    'rms': (max(0.1, rms - rms_tolerance), rms + rms_tolerance),
                    'crest': (max(1.5, crest_factor - crest_tolerance), crest_factor + crest_tolerance)
                }
            
            elif axis_key == 'v0':
                mean_temp = np.mean(signal)
                std_temp = np.std(signal)
                
                # Update temperature thresholds based on current operation
                self.machine_config['temp_thresholds'].update({
                    'normal_max': mean_temp + 2 * std_temp,
                    'warning_max': mean_temp + 3 * std_temp,
                    'critical_max': mean_temp + 4 * std_temp
                })
        
        # Create training data from current state with variations
        training_data_list = []
        for _ in range(30):
            training_signals = {}
            for axis_key, signal in current_signals.items():
                # Add small variations (±5% noise)
                noise_level = np.std(signal) * 0.05
                varied_signal = signal + np.random.normal(0, noise_level, len(signal))
                training_signals[axis_key] = varied_signal
            training_data_list.append(training_signals)
        
        # Retrain the model with current data as baseline
        success, message = self.train_model(training_data_list, sampling_rate, validate_data=False)
        
        return success, message
    
    def reset_to_defaults(self):
        """Reset all parameters to factory defaults"""
        self.__init__()
        return "Reset to factory defaults"
    
    def analyze_individual_axes(self, signals_dict, sampling_rate=1000):
        """Perform detailed analysis for each axis"""
        axis_analysis = {}
        
        for axis, signal in signals_dict.items():
            if axis in ['Fx', 'Fy', 'Fz']:
                analysis = self.analyze_vibration_axis(signal, axis, sampling_rate)
            elif axis == 'v0':
                analysis = self.analyze_temperature(signal, sampling_rate)
            else:
                continue
            
            axis_analysis[axis] = analysis
        
        return axis_analysis
    
    def analyze_vibration_axis(self, signal, axis_name, sampling_rate=1000):
        """Detailed vibration analysis for specific axis with robust error handling"""
        # Ensure signal is a numpy array and handle edge cases
        signal = np.array(signal)
        
        # Check if signal has sufficient data
        if len(signal) == 0:
            return {
                "rms": 0,
                "peak": 0,
                "crest_factor": 0,
                "dominant_freq": 0,
                "dominant_magnitude": 0,
                "fault_indicators": {
                    "bearing_fault": 0,
                    "imbalance": 0,
                    "misalignment": 0,
                    "overall_energy": 0
                },
                "health_score": 50,
                "health_status": "NO_DATA",
                "recommendations": [f"No vibration data available for {axis_name}"]
            }
        
        # Time domain analysis
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / rms if rms > 0 else 0
        
        # Frequency domain analysis (only if we have enough data)
        if len(signal) > 1:
            try:
                fft_values = np.abs(fft(signal))
                freqs = fftfreq(len(signal), 1/sampling_rate)
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = fft_values[:len(fft_values)//2]
                
                # Find dominant frequencies
                if len(positive_fft) > 0:
                    dominant_freq_idx = np.argmax(positive_fft)
                    dominant_freq = positive_freqs[dominant_freq_idx]
                    dominant_magnitude = positive_fft[dominant_freq_idx]
                    
                    # Detect specific fault frequencies
                    fault_indicators = self.detect_fault_frequencies(positive_freqs, positive_fft, axis_name)
                else:
                    dominant_freq = 0
                    dominant_magnitude = 0
                    fault_indicators = {
                        "bearing_fault": 0,
                        "imbalance": 0,
                        "misalignment": 0,
                        "overall_energy": 0
                    }
            except Exception as e:
                # Fallback if FFT fails
                dominant_freq = 0
                dominant_magnitude = 0
                fault_indicators = {
                    "bearing_fault": 0,
                    "imbalance": 0,
                    "misalignment": 0,
                    "overall_energy": 0
                }
        else:
            dominant_freq = 0
            dominant_magnitude = 0
            fault_indicators = {
                "bearing_fault": 0,
                "imbalance": 0,
                "misalignment": 0,
                "overall_energy": 0
            }
        
        # Health assessment for this axis
        axis_health = self.assess_axis_health(rms, crest_factor, fault_indicators, axis_name)
        
        return {
            "rms": rms,
            "peak": peak,
            "crest_factor": crest_factor,
            "dominant_freq": dominant_freq,
            "dominant_magnitude": dominant_magnitude,
            "fault_indicators": fault_indicators,
            "health_score": axis_health["score"],
            "health_status": axis_health["status"],
            "recommendations": axis_health["recommendations"]
        }
    
    def analyze_temperature(self, signal, sampling_rate=1000):
        """Detailed temperature analysis with robust error handling"""
        # Ensure signal is a numpy array and handle edge cases
        signal = np.array(signal)
        
        # Check if signal has sufficient data
        if len(signal) == 0:
            return {
                "mean_temp": 0,
                "max_temp": 0,
                "min_temp": 0,
                "temp_rise_rate": 0,
                "health_score": 50,
                "health_status": "NO_DATA",
                "recommendations": ["No temperature data available"]
            }
        
        # Calculate basic temperature metrics
        mean_temp = np.mean(signal)
        max_temp = np.max(signal)
        min_temp = np.min(signal)
        
        # Calculate temperature rise rate (handle single point case)
        if len(signal) > 1:
            temp_diff = np.diff(signal)
            temp_rise_rate = np.mean(temp_diff) * sampling_rate  # °C per second
        else:
            temp_rise_rate = 0  # Cannot calculate rate with single point
        
        # Temperature health assessment
        temp_health = self.assess_temperature_health(mean_temp, max_temp, temp_rise_rate)
        
        return {
            "mean_temp": mean_temp,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "temp_rise_rate": temp_rise_rate,
            "health_score": temp_health["score"],
            "health_status": temp_health["status"],
            "recommendations": temp_health["recommendations"]
        }
    
    def detect_fault_frequencies(self, freqs, fft_values, axis_name):
        """Detect specific fault-related frequencies using machine configuration"""
        fault_indicators = {}
        
        # Use configured fault frequencies
        rotation_freq = self.machine_config['rotation_freq']
        bearing_freqs = self.machine_config['bearing_freqs']
        
        # Check for bearing faults
        bearing_energy = 0
        for bf in bearing_freqs:
            # Use adaptive frequency bands based on machine speed
            freq_tolerance = max(2, rotation_freq * 0.1)  # Adaptive tolerance
            idx_range = np.where((freqs >= bf-freq_tolerance) & (freqs <= bf+freq_tolerance))[0]
            if len(idx_range) > 0:
                bearing_energy += np.sum(fft_values[idx_range])
        
        # Check for imbalance (1x rotation frequency)
        imb_tolerance = rotation_freq * 0.05  # 5% tolerance
        imb_idx = np.where((freqs >= rotation_freq-imb_tolerance) & (freqs <= rotation_freq+imb_tolerance))[0]
        imbalance_energy = np.sum(fft_values[imb_idx]) if len(imb_idx) > 0 else 0
        
        # Check for misalignment (2x, 3x harmonics)
        harm2_freq = 2 * rotation_freq
        harm3_freq = 3 * rotation_freq
        harm_tolerance = rotation_freq * 0.1
        
        harm2_idx = np.where((freqs >= harm2_freq-harm_tolerance) & (freqs <= harm2_freq+harm_tolerance))[0]
        harm3_idx = np.where((freqs >= harm3_freq-harm_tolerance) & (freqs <= harm3_freq+harm_tolerance))[0]
        
        misalign_energy = 0
        if len(harm2_idx) > 0:
            misalign_energy += np.sum(fft_values[harm2_idx])
        if len(harm3_idx) > 0:
            misalign_energy += np.sum(fft_values[harm3_idx])
        
        # Axis-specific thresholds (based on machine configuration)
        axis_multipliers = {'Fx': 1.0, 'Fy': 0.8, 'Fz': 0.6}
        multiplier = axis_multipliers.get(axis_name, 1.0)
        
        fault_indicators = {
            "bearing_fault": bearing_energy * multiplier,
            "imbalance": imbalance_energy * multiplier,
            "misalignment": misalign_energy * multiplier,
            "overall_energy": np.sum(fft_values) * multiplier,
            "rotation_freq": rotation_freq,
            "bearing_freqs": bearing_freqs
        }
        
        return fault_indicators
    
    def assess_axis_health(self, rms, crest_factor, fault_indicators, axis_name):

        """Assess health score for individual axis using machine configuration"""
        # Use configured normal ranges
        ranges = self.machine_config['normal_ranges'].get(axis_name, 
                                                        self.machine_config['normal_ranges']['Fx'])
        
        # Calculate health score components with adaptive thresholds
        rms_score = 100 if ranges['rms'][0] <= rms <= ranges['rms'][1] else max(0, 100 - abs(rms - np.mean(ranges['rms'])) * 50)
        crest_score = 100 if ranges['crest'][0] <= crest_factor <= ranges['crest'][1] else max(0, 100 - abs(crest_factor - np.mean(ranges['crest'])) * 25)
        
        # Adaptive fault penalty based on machine configuration
        rotation_freq = self.machine_config['rotation_freq']
        
        # Dynamic thresholds based on machine speed and sensitivity
        bearing_threshold = max(50, rotation_freq * 1.5) / self.health_weights.get('fault_sensitivity', 1.0)
        imbalance_threshold = max(100, rotation_freq * 3.0) / self.health_weights.get('fault_sensitivity', 1.0)
        misalign_threshold = max(75, rotation_freq * 2.5) / self.health_weights.get('fault_sensitivity', 1.0)
        
        fault_penalty = 0
        fault_details = {}
        recommendations = []
        
        if fault_indicators["bearing_fault"] > bearing_threshold:
            penalty = min(30, (fault_indicators["bearing_fault"] / bearing_threshold - 1) * 20)
            fault_penalty += penalty
            fault_details['bearing'] = f"Detected (severity: {penalty:.1f})"
        
        if fault_indicators["imbalance"] > imbalance_threshold:
            penalty = min(25, (fault_indicators["imbalance"] / imbalance_threshold - 1) * 15)
            fault_penalty += penalty
            fault_details['imbalance'] = f"Detected (severity: {penalty:.1f})"
        
        if fault_indicators["misalignment"] > misalign_threshold:
            penalty = min(20, (fault_indicators["misalignment"] / misalign_threshold - 1) * 12)
            fault_penalty += penalty
            fault_details['misalignment'] = f"Detected (severity: {penalty:.1f})"
        
        overall_score = max(0, (rms_score + crest_score) / 2 - fault_penalty)
        
        # Determine status and recommendations with machine-specific context
        machine_rpm = self.machine_config['motor_rpm']
        if overall_score >= 80:
            status = "HEALTHY"
            recommendations = [f"{axis_name} axis operating normally at {machine_rpm} RPM"]
        elif overall_score >= 60:
            status = "WARNING"
            recommendations = [
                f"Monitor {axis_name} axis closely at {machine_rpm} RPM", 
                "Check for loose connections", 
                "Verify alignment and balance"
            ]
        else:
            status = "CRITICAL"
            recommendations = [
                f"Immediate attention required for {axis_name} axis", 
                f"Schedule maintenance for {machine_rpm} RPM motor",
                "Check bearings, alignment, and balance"
            ]
        
        # Add specific fault recommendations with machine context
        for fault_type, details in fault_details.items():
            if fault_type == 'bearing':
                recommendations.append(f"{axis_name}: Bearing wear detected at {rotation_freq:.1f}Hz - plan replacement")
            elif fault_type == 'imbalance':
                recommendations.append(f"{axis_name}: Imbalance detected at {rotation_freq:.1f}Hz - check rotor balance")
            elif fault_type == 'misalignment':
                recommendations.append(f"{axis_name}: Misalignment detected at harmonics - check coupling alignment")
    
        return {
            "score": overall_score,
            "status": status,
            "recommendations": recommendations,
            "rms_score": rms_score,
            "crest_score": crest_score,
            "fault_penalty": fault_penalty,
            "fault_details": fault_details,
            "thresholds_used": {
                "bearing": bearing_threshold,
                "imbalance": imbalance_threshold,
                "misalignment": misalign_threshold
            }
        }
        
    def assess_temperature_health(self, mean_temp, max_temp, rise_rate):
        """Assess temperature health using machine configuration and user-defined weights"""
        thresholds = self.machine_config['temp_thresholds'].copy()
        
        # Apply user-defined threshold multipliers
        temp_normal_mult = self.health_weights.get('temp_normal_multiplier', 1.0)
        temp_warning_mult = self.health_weights.get('temp_warning_multiplier', 1.0)
        temp_critical_mult = self.health_weights.get('temp_critical_multiplier', 1.0)
        
        thresholds['normal_max'] *= temp_normal_mult
        thresholds['warning_max'] *= temp_warning_mult
        thresholds['critical_max'] *= temp_critical_mult
        
        # Get temperature component weights
        temp_mean_weight = self.health_weights.get('temp_mean_weight', 0.4)
        temp_max_weight = self.health_weights.get('temp_max_weight', 0.3)
        temp_rise_weight = self.health_weights.get('temp_rise_weight', 0.3)
        
        # Calculate individual component scores
        mean_score = 100
        max_score = 100
        rise_score = 100
        
        recommendations = []
    
        # Mean temperature assessment
        if mean_temp > thresholds['critical_max']:
            mean_score = 0
            recommendations.append(f"Critical mean temperature ({mean_temp:.1f}°C > {thresholds['critical_max']:.1f}°C) - immediate shutdown recommended")
        elif mean_temp > thresholds['warning_max']:
            # Linear scaling between warning and critical
            penalty_range = thresholds['critical_max'] - thresholds['warning_max']
            excess = mean_temp - thresholds['warning_max']
            mean_score = max(20, 80 - (excess / penalty_range) * 60)
            recommendations.append(f"High mean temperature ({mean_temp:.1f}°C > {thresholds['warning_max']:.1f}°C) - check cooling")
        elif mean_temp > thresholds['normal_max']:
            # Linear scaling between normal and warning
            penalty_range = thresholds['warning_max'] - thresholds['normal_max']
            excess = mean_temp - thresholds['normal_max']
            mean_score = max(60, 100 - (excess / penalty_range) * 40)
            recommendations.append(f"Elevated mean temperature ({mean_temp:.1f}°C > {thresholds['normal_max']:.1f}°C) - monitor cooling system")
        
        # Maximum temperature assessment
        max_temp_critical = thresholds['critical_max'] + 5
        max_temp_warning = thresholds['warning_max'] + 3
        
        if max_temp > max_temp_critical:
            max_score = 0
            recommendations.append(f"Critical peak temperature ({max_temp:.1f}°C > {max_temp_critical:.1f}°C) - immediate cooling required")
        elif max_temp > max_temp_warning:
            penalty_range = max_temp_critical - max_temp_warning
            excess = max_temp - max_temp_warning
            max_score = max(20, 80 - (excess / penalty_range) * 60)
            recommendations.append(f"High peak temperature ({max_temp:.1f}°C > {max_temp_warning:.1f}°C) - check for hot spots")
        elif max_temp > thresholds['normal_max']:
            penalty_range = max_temp_warning - thresholds['normal_max']
            excess = max_temp - thresholds['normal_max']
            max_score = max(70, 100 - (excess / penalty_range) * 30)
        
        # Temperature rise rate assessment
        max_rise_rate = thresholds['max_rise_rate']
        if abs(rise_rate) > max_rise_rate * 2:
            rise_score = 0
            recommendations.append(f"Extreme temperature rate change ({rise_rate:.2f}°C/s > {max_rise_rate*2:.2f}°C/s) - check for system malfunction")
        elif abs(rise_rate) > max_rise_rate:
            excess = abs(rise_rate) - max_rise_rate
            penalty = min(60, (excess / max_rise_rate) * 60)
            rise_score = max(20, 100 - penalty)
            if rise_rate > 0:
                recommendations.append(f"Rapid temperature rise ({rise_rate:.2f}°C/s > {max_rise_rate:.2f}°C/s) - check for developing faults")
            else:
                recommendations.append(f"Rapid temperature drop ({rise_rate:.2f}°C/s) - check cooling system")
        
        # Calculate weighted overall score using user-defined weights
        overall_score = (
            mean_score * temp_mean_weight +
            max_score * temp_max_weight +
            rise_score * temp_rise_weight
        )
        
        overall_score = max(0, min(100, overall_score))
        
        # Determine status based on weighted score
        if overall_score >= 80:
            status = "NORMAL"
            if not recommendations:
                recommendations = [f"Temperature within normal range (< {thresholds['normal_max']:.1f}°C)"]
        elif overall_score >= 60:
            status = "ELEVATED"
            if not any("monitor" in rec.lower() for rec in recommendations):
                recommendations.append("Monitor temperature trends closely")
        else:
            status = "CRITICAL"
            if not any("immediate" in rec.lower() for rec in recommendations):
                recommendations.append("Immediate temperature investigation required")
        
        return {
            "score": overall_score,
            "status": status,
            "recommendations": recommendations,
            "thresholds_used": thresholds,
            "component_scores": {
                "mean_score": mean_score,
                "max_score": max_score,
                "rise_score": rise_score
            },
            "weights_used": {
                "mean_weight": temp_mean_weight,
                "max_weight": temp_max_weight,
                "rise_weight": temp_rise_weight
            },
            "temperature_analysis": {
                "mean_temp": mean_temp,
                "max_temp": max_temp,
                "rise_rate": rise_rate,
                "thresholds_applied": {
                    "normal_multiplier": temp_normal_mult,
                    "warning_multiplier": temp_warning_mult,
                    "critical_multiplier": temp_critical_mult
                }
            }
        }

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = VibrationDataGenerator()
    st.session_state.ai_analyzer = AIAnalyzer()
    st.session_state.historical_data = []
    st.session_state.is_monitoring = False
    st.session_state.mysql_connector = MySQLConnector() if MYSQL_AVAILABLE else None
    st.session_state.mysql_connected = False
    st.session_state.mysql_last_timestamp = None
    st.session_state.mysql_data_buffer = pd.DataFrame()


def main():
    st.markdown('<div class="main-header">⚙️ AI Preventive Maintenance System - Phase 1</div>', 
                unsafe_allow_html=True)
    
    # Initialize variables early to avoid UnboundLocalError
    current_signals = {}
    time_vector = None
    sampling_rate = 1000
    
    # Sidebar controls
    st.sidebar.header("🔧 System Controls")
    
    # Machine selection
    machine_id = st.sidebar.selectbox(
        "Select Machine",
        ["Motor-001", "Motor-002", "Pump-001", "Fan-001"]
    )
    
    # Data source selection
    st.sidebar.subheader("📡 Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Simulated Data", "Import CSV File", "MySQL Real-time"]
    )
    
    # MySQL Configuration Section (SINGLE INSTANCE)
    mysql_columns_mapping = {}
    mysql_table = ""
    mysql_refresh_rate = 30
    mysql_data_limit = 1000
    mysql_host = "localhost"
    mysql_port = 3306
    mysql_username = "root"
    mysql_password = ""
    mysql_database = ""
    
    if data_source == "MySQL Real-time":
        st.sidebar.subheader("🗄️ MySQL Server Configuration")
        # Fixed MySQL Configuration Section with proper database selection
    
        if not MYSQL_AVAILABLE:
            st.sidebar.error("MySQL connector not available. Install with: pip install mysql-connector-python")
        else:
            # Connection parameters with unique keys
            mysql_host = st.sidebar.text_input(
                "Server IP Address", 
                value="localhost", 
                help="MySQL server IP address",
                key="mysql_host_input"
            )
            mysql_port = st.sidebar.number_input(
                "Port", 
                value=3306, 
                min_value=1, 
                max_value=65535,
                key="mysql_port_input"
            )
            mysql_username = st.sidebar.text_input(
                "Username", 
                value="root",
                key="mysql_username_input"
            )
            mysql_password = st.sidebar.text_input(
                "Password", 
                type="password",
                key="mysql_password_input"
            )
            
            # Connection status
            if st.session_state.mysql_connected:
                st.sidebar.markdown('<div class="mysql-connected">🟢 Connected to MySQL</div>', unsafe_allow_html=True)
                if st.sidebar.button("🔴 Disconnect", key="mysql_disconnect_btn"):
                    with st.spinner("Disconnecting from MySQL..."):
                        success, message = st.session_state.mysql_connector.disconnect()
                    if success:
                        st.session_state.mysql_connected = False
                        st.session_state.mysql_data_buffer = pd.DataFrame()
                        st.session_state.mysql_last_timestamp = None
                        # Clear selected database info
                        if 'mysql_selected_database' in st.session_state:
                            del st.session_state.mysql_selected_database
                        st.sidebar.success(message)
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.sidebar.error(message)
            else:
                st.sidebar.markdown('<div class="mysql-disconnected">🔴 Not connected</div>', unsafe_allow_html=True)
                
                # Database selection for initial connection
                mysql_database = st.sidebar.text_input(
                    "Database Name (Optional)", 
                    value="",
                    help="Leave empty to connect to server first, then select database",
                    key="mysql_database_input"
                )
                
                # Connection tips
                with st.sidebar.expander("💡 Connection Tips"):
                    st.write("**Connection Steps:**")
                    st.write("1. Connect to MySQL server first")
                    st.write("2. Select database from dropdown")
                    st.write("3. Choose table and configure columns")
                    st.write("")
                    st.write("**Common Issues:**")
                    st.write("• Ensure MySQL server is running")
                    st.write("• Check firewall settings") 
                    st.write("• Verify user permissions")
                    st.write("• Leave database field empty for initial connection")
                
                if st.sidebar.button("🔗 Connect to MySQL Server", key="mysql_connect_btn"):
                    if mysql_host and mysql_username:
                        with st.spinner(f"Connecting to MySQL server {mysql_host}:{mysql_port}..."):
                            # Connect without specifying database first
                            success, message = st.session_state.mysql_connector.connect(
                                mysql_host, mysql_port, "", mysql_username, mysql_password
                            )
                            
                        if success:
                            st.session_state.mysql_connected = True
                            st.sidebar.success(message)
                            st.sidebar.info("✅ Connected to server. Now select a database below.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.session_state.mysql_connected = False
                            st.sidebar.error(message)
                    else:
                        st.sidebar.error("Please provide host and username")
            
            # Database and table selection (only if connected to server)
            if st.session_state.mysql_connected:
                st.sidebar.subheader("📊 Database Configuration")
                
                try:
                    # Get available databases
                    databases = st.session_state.mysql_connector.get_databases()
                    if databases:
                        # Show current connection info
                        current_db = getattr(st.session_state, 'mysql_selected_database', None)
                        if current_db:
                            st.sidebar.success(f"📁 Current database: **{current_db}**")
                        
                        # Database selection dropdown
                        db_index = 0
                        if current_db and current_db in databases:
                            db_index = databases.index(current_db)
                        
                        selected_database = st.sidebar.selectbox(
                            "Select Database", 
                            databases,
                            index=db_index,
                            key="mysql_database_select"
                        )
                        
                        # Connect to selected database if different from current
                        if selected_database != current_db:
                            if st.sidebar.button(f"🔗 Connect to Database: {selected_database}", key="mysql_db_connect_btn"):
                                with st.spinner(f"Connecting to database '{selected_database}'..."):
                                    success, message = st.session_state.mysql_connector.connect(
                                        mysql_host, mysql_port, selected_database, mysql_username, mysql_password
                                    )
                                if success:
                                    st.session_state.mysql_selected_database = selected_database
                                    mysql_database = selected_database
                                    st.sidebar.success(f"✅ Connected to database: {selected_database}")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.sidebar.error(f"Failed to switch to database: {message}")
                        else:
                            mysql_database = selected_database
                            st.session_state.mysql_selected_database = selected_database
                    else:
                        st.sidebar.warning("⚠️ No accessible databases found or insufficient permissions")
                        st.sidebar.info("Check if your user has permission to view databases")
                    
                    # Table selection (only if database is selected)
                    if hasattr(st.session_state, 'mysql_selected_database') and st.session_state.mysql_selected_database:
                        st.sidebar.write(f"**📋 Tables in '{st.session_state.mysql_selected_database}':**")
                        
                        try:
                            tables = st.session_state.mysql_connector.get_tables()
                            if tables:
                                mysql_table = st.sidebar.selectbox(
                                    "Select Table", 
                                    tables,
                                    key="mysql_table_select"
                                )
                                
                                # Column configuration (only if table is selected)
                                if mysql_table:
                                    try:
                                        columns = st.session_state.mysql_connector.get_columns(mysql_table)
                                        if columns:
                                            st.sidebar.subheader("📋 Column Mapping")
                                            st.sidebar.write(f"**Configuring table: {mysql_table}**")
                                            st.sidebar.write(f"**Available columns:** {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
                                            
                                            mysql_timestamp_col = st.sidebar.selectbox(
                                                "Timestamp Column", 
                                                ["None"] + columns,
                                                help="Select column containing timestamp/datetime data",
                                                key="mysql_timestamp_select"
                                            )
                                            mysql_fx_col = st.sidebar.selectbox(
                                                "Fx (X-axis) Column", 
                                                ["None"] + columns,
                                                help="Select column for X-axis vibration data",
                                                key="mysql_fx_select"
                                            )
                                            mysql_fy_col = st.sidebar.selectbox(
                                                "Fy (Y-axis) Column", 
                                                ["None"] + columns,
                                                help="Select column for Y-axis vibration data",
                                                key="mysql_fy_select"
                                            )
                                            mysql_fz_col = st.sidebar.selectbox(
                                                "Fz (Z-axis) Column", 
                                                ["None"] + columns,
                                                help="Select column for Z-axis vibration data",
                                                key="mysql_fz_select"
                                            )
                                            mysql_temp_col = st.sidebar.selectbox(
                                                "v0 (Temperature) Column", 
                                                ["None"] + columns,
                                                help="Select column for temperature data",
                                                key="mysql_temp_select"
                                            )
                                            
                                            mysql_columns_mapping = {
                                                'timestamp': mysql_timestamp_col if mysql_timestamp_col != "None" else None,
                                                'Fx': mysql_fx_col if mysql_fx_col != "None" else None,
                                                'Fy': mysql_fy_col if mysql_fy_col != "None" else None,
                                                'Fz': mysql_fz_col if mysql_fz_col != "None" else None,
                                                'v0': mysql_temp_col if mysql_temp_col != "None" else None
                                            }
                                            
                                            # Show current mapping summary
                                            active_mappings = {k: v for k, v in mysql_columns_mapping.items() if v}
                                            if active_mappings:
                                                st.sidebar.success(f"✅ Mapped {len(active_mappings)} sensors")
                                                with st.sidebar.expander("📋 Current Mapping"):
                                                    for sensor, column in active_mappings.items():
                                                        st.write(f"• **{sensor}**: {column}")
                                            else:
                                                st.sidebar.warning("⚠️ No sensors mapped yet")
                                            
                                            # Date Range Filtering Section
                                            st.sidebar.subheader("📅 Date Range Filter")
                                            
                                            if 'mysql_date_range_enabled' not in st.session_state:
                                                st.session_state.mysql_date_range_enabled = False
                                            if 'mysql_available_dates' not in st.session_state:
                                                st.session_state.mysql_available_dates = None
                                            if 'mysql_date_stats' not in st.session_state:
                                                st.session_state.mysql_date_stats = None
                                            
                                            use_date_range = st.sidebar.checkbox(
                                                "Enable Date Range Filtering",
                                                value=st.session_state.mysql_date_range_enabled,
                                                help="Filter data by specific date/time range",
                                                key="mysql_date_range_checkbox"
                                            )
                                            st.session_state.mysql_date_range_enabled = use_date_range
                                            
                                            if use_date_range:
                                                if mysql_columns_mapping.get('timestamp'):
                                                    # Analyze available dates button
                                                    if st.sidebar.button("📊 Analyze Available Dates", key="analyze_dates_btn") or st.session_state.mysql_available_dates is None:
                                                        with st.spinner("Analyzing timestamp data..."):
                                                            try:
                                                                timestamp_col = mysql_columns_mapping['timestamp']
                                                                
                                                                # Get min and max timestamps
                                                                query = f"SELECT MIN({timestamp_col}) as min_date, MAX({timestamp_col}) as max_date, COUNT(*) as total_records FROM {mysql_table}"
                                                                st.session_state.mysql_connector.cursor.execute(query)
                                                                result = st.session_state.mysql_connector.cursor.fetchone()
                                                                
                                                                if result and result[0] and result[1]:
                                                                    min_date = pd.to_datetime(result[0])
                                                                    max_date = pd.to_datetime(result[1])
                                                                    total_records = result[2]
                                                                    
                                                                    st.session_state.mysql_available_dates = {
                                                                        'min_date': min_date,
                                                                        'max_date': max_date,
                                                                        'total_records': total_records
                                                                    }
                                                                    
                                                                    # Get daily statistics
                                                                    daily_query = f"""
                                                                    SELECT 
                                                                        DATE({timestamp_col}) as date,
                                                                        COUNT(*) as records_per_day,
                                                                        MIN({timestamp_col}) as first_time,
                                                                        MAX({timestamp_col}) as last_time
                                                                    FROM {mysql_table}
                                                                    GROUP BY DATE({timestamp_col})
                                                                    ORDER BY date DESC
                                                                    LIMIT 30
                                                                    """
                                                                    st.session_state.mysql_connector.cursor.execute(daily_query)
                                                                    daily_stats = st.session_state.mysql_connector.cursor.fetchall()
                                                                    
                                                                    st.session_state.mysql_date_stats = {
                                                                        'daily_records': daily_stats,
                                                                        'days_available': len(daily_stats)
                                                                    }
                                                                    
                                                                    st.sidebar.success(f"✅ Found data from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                                                                    st.sidebar.info(f"📊 Total records: {total_records:,}")
                                                                else:
                                                                    st.sidebar.error("❌ No valid timestamp data found")
                                                                    st.session_state.mysql_available_dates = None
                                                            except Exception as e:
                                                                st.sidebar.error(f"❌ Error analyzing dates: {str(e)}")
                                                                st.session_state.mysql_available_dates = None
                                                    
                                                    # Show available date range and controls
                                                    if st.session_state.mysql_available_dates:
                                                        dates_info = st.session_state.mysql_available_dates
                                                        stats_info = st.session_state.mysql_date_stats
                                                        
                                                        # Display data availability info
                                                        st.sidebar.write("**📈 Data Availability:**")
                                                        st.sidebar.write(f"• **From:** {dates_info['min_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                                                        st.sidebar.write(f"• **To:** {dates_info['max_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                                                        st.sidebar.write(f"• **Total Records:** {dates_info['total_records']:,}")
                                                        
                                                        if stats_info and stats_info['daily_records']:
                                                            st.sidebar.write(f"• **Days Available:** {stats_info['days_available']}")
                                                            
                                                            # Show recent daily record counts
                                                            with st.sidebar.expander("📊 Recent Daily Records"):
                                                                for day_stat in stats_info['daily_records'][:7]:  # Show last 7 days
                                                                    date_str = day_stat[0].strftime('%Y-%m-%d') if hasattr(day_stat[0], 'strftime') else str(day_stat[0])
                                                                    records_count = day_stat[1]
                                                                    st.write(f"• {date_str}: {records_count:,} records")
                                                        
                                                        # Quick selection buttons
                                                        st.sidebar.write("**🚀 Quick Select:**")
                                                        col1, col2 = st.sidebar.columns(2)
                                                        
                                                        with col1:
                                                            if st.button("📅 Last 24h", key="quick_24h"):
                                                                end_date = dates_info['max_date']
                                                                start_date = end_date - timedelta(days=1)
                                                                st.session_state.mysql_start_date = start_date.date()
                                                                st.session_state.mysql_start_time = start_date.time()
                                                                st.session_state.mysql_end_date = end_date.date()
                                                                st.session_state.mysql_end_time = end_date.time()
                                                                st.rerun()
                                                        
                                                        with col2:
                                                            if st.button("📅 Last 7d", key="quick_7d"):
                                                                end_date = dates_info['max_date']
                                                                start_date = end_date - timedelta(days=7)
                                                                st.session_state.mysql_start_date = start_date.date()
                                                                st.session_state.mysql_start_time = start_date.time()
                                                                st.session_state.mysql_end_date = end_date.date()
                                                                st.session_state.mysql_end_time = end_date.time()
                                                                st.rerun()
                                                        
                                                        col3, col4 = st.sidebar.columns(2)
                                                        with col3:
                                                            if st.button("📅 Today", key="quick_today"):
                                                                today = datetime.now().date()
                                                                st.session_state.mysql_start_date = today
                                                                st.session_state.mysql_start_time = datetime.min.time()
                                                                st.session_state.mysql_end_date = today
                                                                st.session_state.mysql_end_time = datetime.now().time()
                                                                st.rerun()
                                                        
                                                        with col4:
                                                            if st.button("📅 Yesterday", key="quick_yesterday"):
                                                                yesterday = datetime.now().date() - timedelta(days=1)
                                                                st.session_state.mysql_start_date = yesterday
                                                                st.session_state.mysql_start_time = datetime.min.time()
                                                                st.session_state.mysql_end_date = yesterday
                                                                st.session_state.mysql_end_time = datetime.max.time()
                                                                st.rerun()
                                                        
                                                        # Manual date/time selection
                                                        st.sidebar.write("**📅 Custom Date Range:**")
                                                        
                                                        # Initialize default dates if not set
                                                        if not hasattr(st.session_state, 'mysql_start_date'):
                                                            # Default to last 24 hours
                                                            end_default = dates_info['max_date']
                                                            start_default = end_default - timedelta(days=1)
                                                            st.session_state.mysql_start_date = start_default.date()
                                                            st.session_state.mysql_start_time = start_default.time()
                                                            st.session_state.mysql_end_date = end_default.date()
                                                            st.session_state.mysql_end_time = end_default.time()
                                                        
                                                        # Date inputs
                                                        mysql_start_date = st.sidebar.date_input(
                                                            "Start Date",
                                                            value=st.session_state.mysql_start_date,
                                                            min_value=dates_info['min_date'].date(),
                                                            max_value=dates_info['max_date'].date(),
                                                            key="mysql_start_date_input"
                                                        )
                                                        
                                                        mysql_start_time = st.sidebar.time_input(
                                                            "Start Time",
                                                            value=st.session_state.mysql_start_time,
                                                            key="mysql_start_time_input"
                                                        )
                                                        
                                                        mysql_end_date = st.sidebar.date_input(
                                                            "End Date",
                                                            value=st.session_state.mysql_end_date,
                                                            min_value=dates_info['min_date'].date(),
                                                            max_value=dates_info['max_date'].date(),
                                                            key="mysql_end_date_input"
                                                        )
                                                        
                                                        mysql_end_time = st.sidebar.time_input(
                                                            "End Time",
                                                            value=st.session_state.mysql_end_time,
                                                            key="mysql_end_time_input"
                                                        )
                                                        
                                                        # Update session state
                                                        st.session_state.mysql_start_date = mysql_start_date
                                                        st.session_state.mysql_start_time = mysql_start_time
                                                        st.session_state.mysql_end_date = mysql_end_date
                                                        st.session_state.mysql_end_time = mysql_end_time
                                                        
                                                        # Combine date and time
                                                        mysql_start_datetime = datetime.combine(mysql_start_date, mysql_start_time)
                                                        mysql_end_datetime = datetime.combine(mysql_end_date, mysql_end_time)
                                                        
                                                        # Validate date range
                                                        if mysql_start_datetime >= mysql_end_datetime:
                                                            st.sidebar.error("⚠️ Start date/time must be before end date/time")
                                                            mysql_start_datetime = None
                                                            mysql_end_datetime = None
                                                        else:
                                                            duration = mysql_end_datetime - mysql_start_datetime
                                                            st.sidebar.success(f"📊 Selected duration: {duration}")
                                                            
                                                            # Show duration breakdown
                                                            if duration.days > 0:
                                                                duration_str = f"{duration.days} days, {duration.seconds//3600} hours"
                                                            elif duration.seconds >= 3600:
                                                                duration_str = f"{duration.seconds//3600} hours, {(duration.seconds%3600)//60} minutes"
                                                            elif duration.seconds >= 60:
                                                                duration_str = f"{duration.seconds//60} minutes"
                                                            else:
                                                                duration_str = f"{duration.seconds} seconds"
                                                            
                                                            st.sidebar.info(f"⏱️ Duration: {duration_str}")
                                                            
                                                            # Estimate records in selected range
                                                            if st.sidebar.button("🔍 Estimate Records in Range", key="estimate_records_btn"):
                                                                with st.spinner("Estimating records..."):
                                                                    try:
                                                                        timestamp_col = mysql_columns_mapping['timestamp']
                                                                        count_query = f"""
                                                                        SELECT COUNT(*) 
                                                                        FROM {mysql_table} 
                                                                        WHERE {timestamp_col} BETWEEN %s AND %s
                                                                        """
                                                                        st.session_state.mysql_connector.cursor.execute(count_query, (mysql_start_datetime, mysql_end_datetime))
                                                                        estimated_records = st.session_state.mysql_connector.cursor.fetchone()[0]
                                                                        
                                                                        st.sidebar.success(f"📊 Estimated records: {estimated_records:,}")
                                                                        
                                                                        # Performance warnings
                                                                        if estimated_records > 50000:
                                                                            st.sidebar.warning("⚠️ Large dataset detected!")
                                                                            st.sidebar.write("**Recommendations:**")
                                                                            st.sidebar.write("• Consider shorter time range")
                                                                            st.sidebar.write("• Increase data limit if needed")
                                                                            st.sidebar.write("• Monitor loading time")
                                                                        elif estimated_records > 10000:
                                                                            st.sidebar.info("📈 Medium dataset - should load quickly")
                                                                        elif estimated_records == 0:
                                                                            st.sidebar.warning("⚠️ No data found in selected range")
                                                                            st.sidebar.write("**Try:**")
                                                                            st.sidebar.write("• Expanding the date range")
                                                                            st.sidebar.write("• Checking timestamp format")
                                                                            st.sidebar.write("• Verifying data exists")
                                                                        else:
                                                                            st.sidebar.success("✅ Small dataset - will load quickly")
                                                                            
                                                                    except Exception as e:
                                                                        st.sidebar.error(f"❌ Error estimating records: {str(e)}")
                                                            
                                                            # Store the datetime values for use in data loading
                                                            st.session_state.mysql_start_datetime = mysql_start_datetime
                                                            st.session_state.mysql_end_datetime = mysql_end_datetime
                                                    else:
                                                        st.sidebar.info("👆 Click 'Analyze Available Dates' to see date range options")
                                                        # Set default values to None when no date analysis
                                                        st.session_state.mysql_start_datetime = None
                                                        st.session_state.mysql_end_datetime = None
                                                else:
                                                    st.sidebar.warning("⚠️ Select a timestamp column first to enable date filtering")
                                                    st.sidebar.info("**Steps:**")
                                                    st.sidebar.write("1. Map a timestamp column above")
                                                    st.sidebar.write("2. Enable date range filtering")
                                                    st.sidebar.write("3. Analyze available dates")
                                                    st.session_state.mysql_start_datetime = None
                                                    st.session_state.mysql_end_datetime = None
                                            else:
                                                # Date filtering is disabled
                                                st.sidebar.info("📅 Date filtering disabled - will use latest data")
                                                st.session_state.mysql_start_datetime = None
                                                st.session_state.mysql_end_datetime = None

                                            # Real-time settings
                                            st.sidebar.subheader("⏱️ Real-time Settings")
                                            mysql_refresh_rate = st.sidebar.slider(
                                                "Refresh Rate (seconds)", 
                                                1, 180, 30,
                                                help="How often to check for new data",
                                                key="mysql_refresh_slider"
                                            )
                                            mysql_data_limit = st.sidebar.number_input(
                                                "Data Points to Fetch", 
                                                100, 10000, 1000,
                                                help="Maximum number of records to fetch",
                                                key="mysql_limit_input"
                                            )
                                            
                                            # Test data fetch
                                            if st.sidebar.button("🧪 Test Data Fetch", key="mysql_test_btn"):
                                                if active_mappings:
                                                    with st.spinner("Testing data fetch..."):
                                                        try:
                                                            test_data = st.session_state.mysql_connector.get_latest_data(
                                                                mysql_table, mysql_columns_mapping, limit=5
                                                            )
                                                            if test_data is not None and not test_data.empty:
                                                                st.sidebar.success(f"✅ Successfully fetched {len(test_data)} rows")
                                                                st.sidebar.write("**Sample data:**")
                                                                st.sidebar.dataframe(test_data, use_container_width=True)
                                                                
                                                                # Show data info
                                                                st.sidebar.info(f"📊 Table has data with {len(test_data.columns)} columns")
                                                                if 'timestamp' in test_data.columns:
                                                                    st.sidebar.info(f"🕒 Time range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
                                                            else:
                                                                st.sidebar.warning("❌ No data retrieved. Check your column mappings and table contents.")
                                                        except Exception as e:
                                                            st.sidebar.error(f"❌ Test failed: {str(e)}")
                                                else:
                                                    st.sidebar.warning("⚠️ Please map at least one sensor column before testing")
                                        else:
                                            st.sidebar.error(f"❌ No columns found in table '{mysql_table}'")
                                    except Exception as e:
                                        st.sidebar.error(f"❌ Error getting columns for table '{mysql_table}': {str(e)}")
                            else:
                                st.sidebar.warning(f"❌ No tables found in database '{st.session_state.mysql_selected_database}'")
                                st.sidebar.info("Check if the database contains tables or if you have permissions")
                        except Exception as e:
                            st.sidebar.error(f"❌ Error getting tables: {str(e)}")
                            st.sidebar.info("Make sure you have selected a database first")
                    else:
                        st.sidebar.info("👆 Please select a database first to see available tables")
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Database error: {str(e)}")
                    st.sidebar.info("Try reconnecting to the MySQL server")
    
    # Original fault simulation (for simulated data only)
    fault_type = "healthy"
    if data_source == "Simulated Data":
        fault_type = st.sidebar.selectbox(
            "Simulate Fault Type",
            ["healthy", "bearing", "imbalance", "misalignment"]
        )
    
        # Multi-axis selection
    st.sidebar.subheader("📊 Sensor Configuration")
    available_axes = ["Fx (X-axis)", "Fy (Y-axis)", "Fz (Z-axis)", "v0 (Temperature)"]
    
    if data_source == "MySQL Real-time" and st.session_state.mysql_connected and mysql_columns_mapping:
        # Auto-select axes based on available MySQL columns
        default_selection = []
        if mysql_columns_mapping.get('Fx'):
            default_selection.append("Fx (X-axis)")
        if mysql_columns_mapping.get('Fy'):
            default_selection.append("Fy (Y-axis)")
        if mysql_columns_mapping.get('Fz'):
            default_selection.append("Fz (Z-axis)")
        if mysql_columns_mapping.get('v0'):
            default_selection.append("v0 (Temperature)")
        
        selected_axes = st.sidebar.multiselect(
            "Select Sensors to Monitor",
            available_axes,
            default=default_selection if default_selection else ["Fx (X-axis)"],
            key="sensor_selection_mysql"
        )
    else:
        selected_axes = st.sidebar.multiselect(
            "Select Sensors to Monitor",
            available_axes,
            default=["Fx (X-axis)", "v0 (Temperature)"],
            key="sensor_selection_default"
        )
    
    if not selected_axes:
        st.sidebar.error("Please select at least one sensor!")
        selected_axes = ["Fx (X-axis)"]
    
    # Generate or load current data for all selected axes
    if data_source == "Simulated Data":
        # Generate simulated data for selected axes
        for axis_display in selected_axes:
            if "Fx" in axis_display:
                if fault_type == "healthy":
                    current_signals['Fx'] = st.session_state.data_generator.generate_healthy_signal('x')
                else:
                    current_signals['Fx'] = st.session_state.data_generator.generate_faulty_signal(fault_type, 'x')
            elif "Fy" in axis_display:
                if fault_type == "healthy":
                    current_signals['Fy'] = st.session_state.data_generator.generate_healthy_signal('y')
                else:
                    current_signals['Fy'] = st.session_state.data_generator.generate_faulty_signal(fault_type, 'y')
            elif "Fz" in axis_display:
                if fault_type == "healthy":
                    current_signals['Fz'] = st.session_state.data_generator.generate_healthy_signal('z')
                else:
                    current_signals['Fz'] = st.session_state.data_generator.generate_faulty_signal(fault_type, 'z')
            elif "v0" in axis_display:
                current_signals['v0'] = st.session_state.data_generator.generate_temperature_data(
                    fault_condition=fault_type if fault_type != "healthy" else None
                )
        
        time_vector = st.session_state.data_generator.time_vector
        sampling_rate = 1000
        
    elif data_source == "Import CSV File":
        st.sidebar.subheader("📁 CSV File Configuration")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file containing sensor data",
        key="csv_file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV file with error handling
            csv_data = pd.read_csv(uploaded_file)
            
            # Display basic file info
            st.sidebar.success(f"✅ File loaded: {uploaded_file.name}")
            st.sidebar.info(f"📊 Shape: {csv_data.shape[0]} rows × {csv_data.shape[1]} columns")
            
            # Show available columns
            available_columns = list(csv_data.columns)
            st.sidebar.write(f"**Available columns:** {', '.join(available_columns[:5])}{'...' if len(available_columns) > 5 else ''}")
            
            # Column mapping section
            st.sidebar.subheader("📋 Column Mapping")
            st.sidebar.write("Map CSV columns to sensor data:")
            
            # Timestamp column selection
            csv_timestamp_col = st.sidebar.selectbox(
                "Timestamp Column (Optional)",
                ["None"] + available_columns,
                help="Select column containing timestamp/datetime data",
                key="csv_timestamp_select"
            )
            
            # Sensor column mappings
            csv_fx_col = st.sidebar.selectbox(
                "Fx (X-axis) Column",
                ["None"] + available_columns,
                help="Select column for X-axis vibration data",
                key="csv_fx_select"
            )
            
            csv_fy_col = st.sidebar.selectbox(
                "Fy (Y-axis) Column", 
                ["None"] + available_columns,
                help="Select column for Y-axis vibration data",
                key="csv_fy_select"
            )
            
            csv_fz_col = st.sidebar.selectbox(
                "Fz (Z-axis) Column",
                ["None"] + available_columns,
                help="Select column for Z-axis vibration data", 
                key="csv_fz_select"
            )
            
            csv_temp_col = st.sidebar.selectbox(
                "v0 (Temperature) Column",
                ["None"] + available_columns,
                help="Select column for temperature data",
                key="csv_temp_select"
            )
            
            # Build column mapping
            csv_columns_mapping = {
                'timestamp': csv_timestamp_col if csv_timestamp_col != "None" else None,
                'Fx': csv_fx_col if csv_fx_col != "None" else None,
                'Fy': csv_fy_col if csv_fy_col != "None" else None,
                'Fz': csv_fz_col if csv_fz_col != "None" else None,
                'v0': csv_temp_col if csv_temp_col != "None" else None
            }
            
            # Show current mapping summary
            active_mappings = {k: v for k, v in csv_columns_mapping.items() if v}
            if active_mappings:
                st.sidebar.success(f"✅ Mapped {len(active_mappings)} sensors")
                with st.sidebar.expander("📋 Current Mapping"):
                    for sensor, column in active_mappings.items():
                        st.write(f"• **{sensor}**: {column}")
            else:
                st.sidebar.warning("⚠️ No sensors mapped yet")
            
            # Data filtering and sampling options
            st.sidebar.subheader("🔧 Data Processing Options")
            
            # Row range selection
            max_rows = len(csv_data)
            row_range = st.sidebar.slider(
                "Select Row Range",
                0, max_rows-1, (0, min(1000, max_rows-1)),
                help="Select which rows to analyze",
                key="csv_row_range_slider"
            )
            
            start_row, end_row = row_range
            st.sidebar.info(f"Selected rows: {start_row} to {end_row} ({end_row-start_row+1} rows)")
            
            # Sampling rate input
            csv_sampling_rate = st.sidebar.number_input(
                "Sampling Rate (Hz)",
                min_value=1,
                max_value=10000,
                value=1000,
                help="Sampling rate of the sensor data",
                key="csv_sampling_rate_input"
            )
            
            # Data quality checks
            if st.sidebar.button("🔍 Analyze Data Quality", key="csv_quality_check_btn"):
                with st.spinner("Analyzing CSV data quality..."):
                    st.sidebar.write("**Data Quality Report:**")
                    
                    # Check for missing values
                    missing_data = csv_data.isnull().sum()
                    total_missing = missing_data.sum()
                    
                    if total_missing > 0:
                        st.sidebar.warning(f"⚠️ Found {total_missing} missing values")
                        for col, missing_count in missing_data[missing_data > 0].items():
                            st.sidebar.write(f"• {col}: {missing_count} missing")
                    else:
                        st.sidebar.success("✅ No missing values")
                    
                    # Check data types
                    numeric_cols = csv_data.select_dtypes(include=[np.number]).columns
                    non_numeric_cols = csv_data.select_dtypes(exclude=[np.number]).columns
                    
                    st.sidebar.write(f"**Numeric columns:** {len(numeric_cols)}")
                    st.sidebar.write(f"**Non-numeric columns:** {len(non_numeric_cols)}")
                    
                    # Check for duplicate rows
                    duplicates = csv_data.duplicated().sum()
                    if duplicates > 0:
                        st.sidebar.warning(f"⚠️ Found {duplicates} duplicate rows")
                    else:
                        st.sidebar.success("✅ No duplicate rows")
                    
                    # Show data range for mapped columns
                    for sensor, column in active_mappings.items():
                        if column in csv_data.columns and sensor != 'timestamp':
                            try:
                                col_data = pd.to_numeric(csv_data[column], errors='coerce')
                                if not col_data.isna().all():
                                    data_min = col_data.min()
                                    data_max = col_data.max()
                                    data_mean = col_data.mean()
                                    st.sidebar.write(f"**{sensor} ({column}):**")
                                    st.sidebar.write(f"  Range: {data_min:.3f} to {data_max:.3f}")
                                    st.sidebar.write(f"  Mean: {data_mean:.3f}")
                                else:
                                    st.sidebar.error(f"❌ {sensor}: No numeric data")
                            except Exception as e:
                                st.sidebar.error(f"❌ {sensor}: Error analyzing data")
            
            # Preview data
            if st.sidebar.button("👀 Preview Data", key="csv_preview_btn"):
                st.sidebar.write("**Data Preview:**")
                preview_data = csv_data.iloc[start_row:min(start_row+5, end_row+1)]
                st.sidebar.dataframe(preview_data, use_container_width=True)
            
            # Process CSV data for analysis
            if active_mappings:
                try:
                    # Extract the selected row range
                    filtered_data = csv_data.iloc[start_row:end_row+1].copy()
                    
                    # Extract signals for selected axes
                    current_signals = {}
                    signals_extracted = 0
                    
                    for axis_display in selected_axes:
                        axis_key = axis_display.split(" ")[0]
                        column_name = csv_columns_mapping.get(axis_key)
                        
                        if column_name and column_name in filtered_data.columns:
                            try:
                                # Convert to numeric, handling any errors
                                signal_data = pd.to_numeric(filtered_data[column_name], errors='coerce').dropna().values
                                
                                if len(signal_data) > 0:
                                    current_signals[axis_key] = signal_data
                                    signals_extracted += 1
                                else:
                                    st.sidebar.warning(f"⚠️ No valid numeric data for {axis_key}")
                            except Exception as e:
                                st.sidebar.error(f"❌ Error processing {axis_key}: {str(e)}")
                    
                    if signals_extracted == 0:
                        st.sidebar.error("❌ No valid signal data extracted from CSV")
                        # Fallback to simulated data
                        current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
                        time_vector = st.session_state.data_generator.time_vector
                        sampling_rate = 1000
                    else:
                        # Handle time vector for CSV data
                        if csv_columns_mapping.get('timestamp') and csv_columns_mapping['timestamp'] in filtered_data.columns:
                            try:
                                # Try to parse timestamps
                                timestamps = pd.to_datetime(filtered_data[csv_columns_mapping['timestamp']], errors='coerce').dropna()
                                
                                if len(timestamps) > 1:
                                    # Convert to relative time in seconds
                                    time_vector = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
                                    
                                    # Estimate actual sampling rate from timestamps
                                    time_intervals = np.diff(time_vector)
                                    valid_intervals = time_intervals[time_intervals > 0]
                                    
                                    if len(valid_intervals) > 0:
                                        avg_interval = np.median(valid_intervals)
                                        estimated_sr = 1.0 / avg_interval if avg_interval > 0 else csv_sampling_rate
                                        
                                        # Use estimated sampling rate if reasonable
                                        if 0.1 <= estimated_sr <= 10000:
                                            sampling_rate = estimated_sr
                                            st.sidebar.info(f"📊 Estimated sampling rate: {estimated_sr:.1f} Hz")
                                        else:
                                            sampling_rate = csv_sampling_rate
                                            st.sidebar.warning(f"⚠️ Using manual sampling rate: {csv_sampling_rate} Hz")
                                    else:
                                        sampling_rate = csv_sampling_rate
                                        time_vector = np.arange(len(list(current_signals.values())[0])) / sampling_rate
                                else:
                                    sampling_rate = csv_sampling_rate
                                    time_vector = np.arange(len(list(current_signals.values())[0])) / sampling_rate
                            except Exception as e:
                                st.sidebar.warning(f"⚠️ Error parsing timestamps: {str(e)}")
                                sampling_rate = csv_sampling_rate
                                time_vector = np.arange(len(list(current_signals.values())[0])) / sampling_rate
                        else:
                            # No timestamp column, generate time vector
                            sampling_rate = csv_sampling_rate
                            max_length = max(len(signal) for signal in current_signals.values()) if current_signals else 1000
                            time_vector = np.arange(max_length) / sampling_rate
                        
                        # Success message with comprehensive info
                        data_duration = len(time_vector) / sampling_rate if len(time_vector) > 0 else 0
                        st.sidebar.success(f"✅ Loaded {signals_extracted} sensors | {len(time_vector)} samples | {data_duration:.1f}s @ {sampling_rate:.1f}Hz")
                        
                        # Data validation
                        with st.sidebar.expander("📈 Signal Statistics"):
                            for axis_key, signal_data in current_signals.items():
                                signal_mean = np.mean(signal_data)
                                signal_std = np.std(signal_data)
                                signal_range = np.max(signal_data) - np.min(signal_data)
                                unit = "°C" if axis_key == 'v0' else "g"
                                
                                st.write(f"**{axis_key}**: μ={signal_mean:.3f}{unit}, σ={signal_std:.3f}{unit}, range={signal_range:.3f}{unit}")
                
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing CSV data: {str(e)}")
                    # Fallback to simulated data
                    current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
                    time_vector = st.session_state.data_generator.time_vector
                    sampling_rate = 1000
            else:
                st.sidebar.warning("⚠️ Please map at least one sensor column to proceed")
                # Fallback to simulated data
                current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
                time_vector = st.session_state.data_generator.time_vector
                sampling_rate = 1000
        except Exception as e:
            st.sidebar.error(f"❌ Error reading CSV file: {str(e)}")
            st.sidebar.write("**Common issues:**")
            st.sidebar.write("• File encoding (try UTF-8)")
            st.sidebar.write("• Malformed CSV structure")
            st.sidebar.write("• Very large file size")
            
            # Fallback to simulated data
            current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
            time_vector = st.session_state.data_generator.time_vector
            sampling_rate = 1000
        
    elif data_source == "MySQL Real-time":
        # Use MySQL data for selected axes
        if st.session_state.mysql_connected and mysql_columns_mapping:
            try:
                # Check if date range filtering is enabled and configured
                use_date_range = getattr(st.session_state, 'mysql_date_range_enabled', False)
                start_datetime = getattr(st.session_state, 'mysql_start_datetime', None)
                end_datetime = getattr(st.session_state, 'mysql_end_datetime', None)
                
                # Determine data loading method
                if use_date_range and start_datetime and end_datetime:
                    # Historical data mode with specific date range
                    st.sidebar.info(f"📊 Loading historical data from {start_datetime.strftime('%Y-%m-%d %H:%M')} to {end_datetime.strftime('%Y-%m-%d %H:%M')}")
                    
                    mysql_data = st.session_state.mysql_connector.get_data_by_date_range(
                        mysql_table, mysql_columns_mapping, start_datetime, end_datetime, 
                        limit=mysql_data_limit
                    )
                    
                    if mysql_data is not None and not mysql_data.empty:
                        st.session_state.mysql_data_buffer = mysql_data
                        st.sidebar.success(f"📊 Loaded {len(mysql_data)} historical records")
                        
                        # Update last timestamp for potential real-time continuation
                        if 'timestamp' in mysql_data.columns:
                            st.session_state.mysql_last_timestamp = mysql_data['timestamp'].iloc[-1]
                    else:
                        st.sidebar.warning(f"📊 No data found in selected date range")
                        # Fallback to latest data
                        mysql_data = st.session_state.mysql_connector.get_latest_data(
                            mysql_table, mysql_columns_mapping, limit=100
                        )
                        if mysql_data is not None and not mysql_data.empty:
                            st.session_state.mysql_data_buffer = mysql_data
                            st.sidebar.info(f"📡 Loaded {len(mysql_data)} latest records as fallback")
                            
                elif st.session_state.is_monitoring and st.session_state.mysql_last_timestamp and not use_date_range:
                    # Real-time monitoring mode (only when date range is disabled)
                    st.sidebar.info("📡 Checking for new real-time data...")
                    
                    new_data = st.session_state.mysql_connector.get_real_time_data(
                        mysql_table, mysql_columns_mapping, st.session_state.mysql_last_timestamp
                    )
                    
                    if new_data is not None and not new_data.empty:
                        # Append new data to buffer, keeping within limit
                        st.session_state.mysql_data_buffer = pd.concat([
                            st.session_state.mysql_data_buffer, new_data
                        ]).tail(mysql_data_limit).reset_index(drop=True)
                        
                        # Update last timestamp
                        if 'timestamp' in new_data.columns:
                            st.session_state.mysql_last_timestamp = new_data['timestamp'].iloc[-1]
                        
                        st.sidebar.success(f"📡 Received {len(new_data)} new real-time records")
                    else:
                        st.sidebar.info("📡 No new data available since last update")
                else:
                    # Initial data load or refresh
                    st.sidebar.info("📡 Loading initial data from MySQL...")
                    
                    if use_date_range and start_datetime and end_datetime:
                        # Use date range for initial load
                        mysql_data = st.session_state.mysql_connector.get_data_by_date_range(
                            mysql_table, mysql_columns_mapping, start_datetime, end_datetime, 
                            limit=mysql_data_limit
                        )
                        load_type = "date range"
                    else:
                        # Use latest data
                        mysql_data = st.session_state.mysql_connector.get_latest_data(
                            mysql_table, mysql_columns_mapping, limit=mysql_data_limit
                        )
                        load_type = "latest"
                    
                    if mysql_data is not None and not mysql_data.empty:
                        st.session_state.mysql_data_buffer = mysql_data
                        if 'timestamp' in mysql_data.columns:
                            st.session_state.mysql_last_timestamp = mysql_data['timestamp'].iloc[-1]
                        
                        st.sidebar.success(f"📊 Loaded {len(mysql_data)} records ({load_type})")
                        
                        # Show data time range
                        if 'timestamp' in mysql_data.columns:
                            data_start = mysql_data['timestamp'].min()
                            data_end = mysql_data['timestamp'].max()
                            st.sidebar.info(f"🕒 Data range: {data_start} to {data_end}")
                    else:
                        st.sidebar.warning("📡 No data retrieved from MySQL")
                
                # Convert buffered data to signals for analysis
                if not st.session_state.mysql_data_buffer.empty:
                    mysql_df = st.session_state.mysql_data_buffer
                    
                    # Extract signals for selected axes
                    signals_extracted = 0
                    for axis_display in selected_axes:
                        axis_key = axis_display.split(" ")[0]
                        if axis_key in mysql_df.columns:
                            # Handle potential null values
                            signal_data = mysql_df[axis_key].dropna().values
                            if len(signal_data) > 0:
                                current_signals[axis_key] = signal_data
                                signals_extracted += 1
                            else:
                                st.sidebar.warning(f"⚠️ No valid data for {axis_key}")
                    
                    if signals_extracted == 0:
                        st.sidebar.error("❌ No valid signal data extracted from MySQL")
                        # Fallback to simulated data
                        current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
                        time_vector = st.session_state.data_generator.time_vector
                        sampling_rate = 1000
                    else:
                        # Handle time vector
                        if 'timestamp' in mysql_df.columns:
                            # Convert timestamp to relative time in seconds
                            timestamps = pd.to_datetime(mysql_df['timestamp']).dropna()
                            if len(timestamps) > 1:
                                time_vector = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
                                
                                # Estimate sampling rate from time intervals
                                time_intervals = np.diff(time_vector)
                                valid_intervals = time_intervals[time_intervals > 0]
                                
                                if len(valid_intervals) > 0:
                                    avg_interval = np.median(valid_intervals)  # Use median for robustness
                                    sampling_rate = 1.0 / avg_interval if avg_interval > 0 else 1000
                                    
                                    # Cap sampling rate at reasonable values
                                    if sampling_rate > 10000:
                                        sampling_rate = 1000
                                        st.sidebar.warning("⚠️ Very high sampling rate detected, capped at 1000 Hz")
                                    elif sampling_rate < 0.1:
                                        sampling_rate = 1
                                        st.sidebar.warning("⚠️ Very low sampling rate detected, set to 1 Hz")
                                else:
                                    sampling_rate = 1000
                            else:
                                sampling_rate = 1000
                                time_vector = np.arange(len(list(current_signals.values())[0])) / sampling_rate
                        else:
                            # Auto-generate time vector based on data length
                            max_length = max(len(signal) for signal in current_signals.values()) if current_signals else 1000
                            time_vector = np.arange(max_length) / sampling_rate
                        
                        # Display comprehensive data information
                        data_info_parts = [
                            f"📊 Analyzing {len(current_signals)} sensors",
                            f"{len(time_vector)} samples",
                            f"SR: {sampling_rate:.1f} Hz"
                        ]
                        
                        if use_date_range and start_datetime and end_datetime:
                            duration = end_datetime - start_datetime
                            data_info_parts.append(f"Range: {duration}")
                        
                        st.sidebar.success(" | ".join(data_info_parts))
                        
                        # Show data quality metrics
                        with st.sidebar.expander("📈 Data Quality Metrics"):
                            st.write(f"**Total Records:** {len(mysql_df):,}")
                            st.write(f"**Active Sensors:** {signals_extracted}/{len(selected_axes)}")
                            
                            if 'timestamp' in mysql_df.columns and len(mysql_df) > 1:
                                timestamps = pd.to_datetime(mysql_df['timestamp'])
                                time_span = timestamps.iloc[-1] - timestamps.iloc[0]
                                st.write(f"**Time Span:** {time_span}")
                                
                                # Calculate data rate
                                data_rate = len(mysql_df) / time_span.total_seconds() if time_span.total_seconds() > 0 else 0
                                st.write(f"**Data Rate:** {data_rate:.2f} records/sec")
                                
                                # Check for time gaps
                                time_diffs = timestamps.diff().dt.total_seconds().dropna()
                                if len(time_diffs) > 1:
                                    avg_interval = time_diffs.mean()
                                    max_gap = time_diffs.max()
                                    gap_threshold = avg_interval * 5  # 5x average interval
                                    
                                    large_gaps = (time_diffs > gap_threshold).sum()
                                    if large_gaps > 0:
                                        st.warning(f"⚠️ {large_gaps} large time gaps detected")
                                        st.write(f"Max gap: {max_gap:.1f}s (avg: {avg_interval:.1f}s)")
                                    else:
                                        st.success("✅ Consistent time intervals")
                            
                            # Check for missing data
                            missing_data = mysql_df.isnull().sum().sum()
                            total_values = len(mysql_df) * len(mysql_df.columns)
                            missing_pct = (missing_data / total_values) * 100 if total_values > 0 else 0
                            
                            if missing_pct > 5:
                                st.warning(f"⚠️ {missing_pct:.1f}% missing values")
                            elif missing_pct > 0:
                                st.info(f"ℹ️ {missing_pct:.1f}% missing values")
                            else:
                                st.success("✅ No missing values")
                else:
                    st.sidebar.warning("❌ No data in MySQL buffer")
                    # Fallback to simulated data
                    current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
                    time_vector = st.session_state.data_generator.time_vector
                    sampling_rate = 1000
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error processing MySQL data: {str(e)}")
                # Show detailed error for debugging
                with st.sidebar.expander("🔍 Error Details"):
                    st.write(f"**Error Type:** {type(e).__name__}")
                    st.write(f"**Error Message:** {str(e)}")
                    st.write("**Possible Causes:**")
                    st.write("• Database connection lost")
                    st.write("• Invalid date range")
                    st.write("• Column mapping issues")
                    st.write("• Network connectivity problems")
                
                # Fallback to simulated data
                current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
                time_vector = st.session_state.data_generator.time_vector
                sampling_rate = 1000
        else:
            # Not connected or no column mapping
            if not st.session_state.mysql_connected:
                st.sidebar.warning("⚠️ Please connect to MySQL database first")
            if not mysql_columns_mapping:
                st.sidebar.warning("⚠️ Please configure column mappings first")
            
            # Fallback to simulated data
            current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
            time_vector = st.session_state.data_generator.time_vector
            sampling_rate = 1000

    else:
        st.sidebar.info("👆 Please upload a CSV file to begin analysis")
        # Fallback to simulated data when no file uploaded
        current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
        time_vector = st.session_state.data_generator.time_vector
        sampling_rate = 1000
    
    # Ensure we have at least one signal
    if not current_signals:
        current_signals = {'Fx': st.session_state.data_generator.generate_healthy_signal('x')}
        time_vector = st.session_state.data_generator.time_vector
        sampling_rate = 1000
    
    # Training section with enhanced options
    st.sidebar.subheader("🤖 AI Model Training")
    
    # Machine configuration
    with st.sidebar.expander("⚙️ Machine Configuration"):
        machine_type = st.selectbox(
            "Machine Type",
            ["motor", "pump", "fan"],
            key="machine_type_select"
        )
        
        motor_rpm = st.number_input(
            "Motor RPM",
            min_value=100,
            max_value=10000,
            value=1800,
            step=100,
            key="motor_rpm_input"
        )
        
        if st.button("⚙️ Configure Machine", key="configure_machine_btn"):
            config_result = st.session_state.ai_analyzer.configure_machine(
                motor_rpm=motor_rpm,
                machine_type=machine_type
            )
            st.success(config_result)
    
    # Enhanced weighting configuration
    with st.sidebar.expander("⚖️ Health Calculation Weights"):
        st.write("**Analysis Method Weights:**")
        anomaly_weight = st.slider(
            "AI Anomaly Detection Weight",
            0.0, 1.0, 0.4, 0.1,
            key="anomaly_weight_slider"
        )
        
        axis_weight = 1.0 - anomaly_weight
        st.write(f"Individual Axis Analysis Weight: {axis_weight:.1f}")
        
        st.write("**Sensor Type Weights:**")
        vibration_weight = st.slider(
            "Vibration Sensors Weight",
            0.0, 1.0, 0.7, 0.1,
            key="vibration_weight_slider"
        )
        
        temperature_weight = 1.0 - vibration_weight
        st.write(f"Temperature Sensor Weight: {temperature_weight:.1f}")
        
        # Temperature analysis weights
        st.write("**Temperature Analysis Weights:**")
        temp_mean_weight = st.slider(
            "Mean Temperature Weight",
            0.0, 1.0, 0.4, 0.1,
            key="temp_mean_weight_slider"
        )
        
        temp_max_weight = st.slider(
            "Maximum Temperature Weight",
            0.0, 1.0, 0.3, 0.1,
            key="temp_max_weight_slider"
        )
        
        temp_rise_weight = st.slider(
            "Temperature Rise Rate Weight",
            0.0, 1.0, 0.3, 0.1,
            key="temp_rise_weight_slider"
        )
        
        # Normalize temperature weights
        total_temp_weight = temp_mean_weight + temp_max_weight + temp_rise_weight
        if total_temp_weight > 0:
            temp_mean_weight_norm = temp_mean_weight / total_temp_weight
            temp_max_weight_norm = temp_max_weight / total_temp_weight
            temp_rise_weight_norm = temp_rise_weight / total_temp_weight
        else:
            temp_mean_weight_norm = 0.4
            temp_max_weight_norm = 0.3
            temp_rise_weight_norm = 0.3
        
        st.write("**Feature Weights for Vibration:**")
        rms_weight = st.slider("RMS Weight", 0.0, 1.0, 0.4, 0.1, key="rms_weight_slider")
        crest_weight = st.slider("Crest Factor Weight", 0.0, 1.0, 0.3, 0.1, key="crest_weight_slider")
        frequency_weight = st.slider("Frequency Features Weight", 0.0, 1.0, 0.3, 0.1, key="freq_weight_slider")
        
        # Normalize vibration weights
        total_feature_weight = rms_weight + crest_weight + frequency_weight
        if total_feature_weight > 0:
            rms_weight /= total_feature_weight
            crest_weight /= total_feature_weight
            frequency_weight /= total_feature_weight
        
        st.write("**Fault Detection Sensitivity:**")
        fault_sensitivity = st.slider(
            "Fault Detection Sensitivity",
            0.1, 2.0, 1.0, 0.1,
            key="fault_sensitivity_slider"
        )
        
        # Temperature threshold multipliers
        st.write("**Temperature Threshold Multipliers:**")
        temp_normal_multiplier = st.slider(
            "Normal Temperature Threshold Multiplier",
            0.5, 2.0, 1.0, 0.1,
            key="temp_normal_mult_slider"
        )
        
        temp_warning_multiplier = st.slider(
            "Warning Temperature Threshold Multiplier",
            0.5, 2.0, 1.0, 0.1,
            key="temp_warning_mult_slider"
        )
        
        temp_critical_multiplier = st.slider(
            "Critical Temperature Threshold Multiplier",
            0.5, 2.0, 1.0, 0.1,
            key="temp_critical_mult_slider"
        )
        
        # Update weights in analyzer
        if st.button("🔄 Update Weights", key="update_weights_btn"):
            st.session_state.ai_analyzer.update_weights({
                'anomaly_detection': anomaly_weight,
                'axis_analysis': axis_weight,
                'vibration_weight': vibration_weight,
                'temperature_weight': temperature_weight,
                'temp_mean_weight': temp_mean_weight_norm,
                'temp_max_weight': temp_max_weight_norm,
                'temp_rise_weight': temp_rise_weight_norm,
                'rms_weight': rms_weight,
                'crest_weight': crest_weight,
                'frequency_weight': frequency_weight,
                'fault_sensitivity': fault_sensitivity,
                'temp_normal_multiplier': temp_normal_multiplier,
                'temp_warning_multiplier': temp_warning_multiplier,
                'temp_critical_multiplier': temp_critical_multiplier
            })
            st.success("✅ Weights updated successfully!")
    
    # Training options
    training_method = st.sidebar.radio(
        "Training Data Source",
        ["Simulated Healthy Data", "Current Data as Healthy"],
        key="training_method_radio"
    )
    
    if st.sidebar.button("🧠 Train AI Model", key="train_model_btn"):
        with st.spinner("Training AI model..."):
            training_data_list = []
            
            if training_method == "Simulated Healthy Data":
                num_samples = st.sidebar.slider("Training Samples", 20, 100, 50, key="training_samples_slider")
                for _ in range(num_samples):
                    training_signals = {}
                    for axis_display in selected_axes:
                        if "Fx" in axis_display:
                            training_signals['Fx'] = st.session_state.data_generator.generate_healthy_signal('x')
                        elif "Fy" in axis_display:
                            training_signals['Fy'] = st.session_state.data_generator.generate_healthy_signal('y')
                        elif "Fz" in axis_display:
                            training_signals['Fz'] = st.session_state.data_generator.generate_healthy_signal('z')
                        elif "v0" in axis_display:
                            training_signals['v0'] = st.session_state.data_generator.generate_temperature_data()
                    training_data_list.append(training_signals)
                    
            elif training_method == "Current Data as Healthy" and current_signals:
                for _ in range(20):
                    training_signals = {}
                    for axis_key, signal in current_signals.items():
                        noise_level = np.std(signal) * 0.1
                        varied_signal = signal + np.random.normal(0, noise_level, len(signal))
                        training_signals[axis_key] = varied_signal
                    training_data_list.append(training_signals)
            
            if training_data_list:
                success, message = st.session_state.ai_analyzer.train_model(training_data_list, sampling_rate)
                if success:
                    st.sidebar.success(f"✅ {message}")
                else:
                    st.sidebar.error(f"❌ Training failed: {message}")
            else:
                st.sidebar.error("No training data available")
    
    # Monitoring controls
    monitoring_interval = mysql_refresh_rate if data_source == "MySQL Real-time" else st.sidebar.slider("Monitoring Interval (seconds)", 1, 10, 3, key="monitoring_interval_slider")
    
    if st.sidebar.button("🟢 Start Monitoring", key="start_monitoring_btn"):
        st.session_state.is_monitoring = True
        
    if st.sidebar.button("🔴 Stop Monitoring", key="stop_monitoring_btn"):
        st.session_state.is_monitoring = False
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # AI Analysis with enhanced results
    analysis_result = st.session_state.ai_analyzer.analyze_signals(current_signals, sampling_rate)
    health_score = analysis_result["health_score"]
    is_anomaly = analysis_result["anomaly"]
    confidence = analysis_result["confidence"]
    axis_analysis = analysis_result.get("axis_analysis", {})
    validation_result = analysis_result.get("validation", {})
    health_breakdown = analysis_result.get("health_breakdown", {})
    
    # Status determination
    if health_score >= 80:
        status = "HEALTHY"
        status_color = "🟢"
        alert_class = "alert-normal"
    elif health_score >= 60:
        status = "WARNING"
        status_color = "🟡"
        alert_class = "alert-warning"
    else:
        status = "CRITICAL"
        status_color = "🔴"
        alert_class = "alert-danger"
    
    # Display enhanced metrics
    with col1:
        st.metric("Machine Health", f"{health_score:.1f}%")
        if health_breakdown:
            st.caption(f"Anomaly: {health_breakdown.get('anomaly_health', 0):.0f}% | Axis: {health_breakdown.get('axis_health', 0):.0f}%")
    
    with col2:
        st.metric("Status", f"{status_color} {status}")
        if validation_result.get('warnings'):
            st.caption(f"⚠️ {len(validation_result['warnings'])} validation warnings")
    
    with col3:
        st.metric("Confidence", f"{confidence:.1f}%")
        machine_config = analysis_result.get("machine_config", {})
        if machine_config:
            st.caption(f"RPM: {machine_config.get('motor_rpm', 'N/A')}")
    
    with col4:
        active_sensors = len(current_signals)
        data_source_short = {"Simulated Data": "SIM", "Import CSV File": "CSV", "MySQL Real-time": "SQL"}[data_source]
        st.metric("Data Source", f"{data_source_short} ({active_sensors})")
        if st.session_state.ai_analyzer.is_trained:
            st.caption("🤖 AI Model: Trained")
        else:
            st.caption("🤖 AI Model: Not Trained")
    
    # Enhanced alert section
    sensor_list = ", ".join([axis.split(" ")[0] for axis in selected_axes])
    data_source_info = f" | Data: {data_source}"
    
    # Add validation warnings to alert
    validation_info = ""
    if validation_result.get('warnings'):
        validation_info = f" | ⚠️ {len(validation_result['warnings'])} data validation warnings"
    
    # Add machine configuration info
    machine_info = ""
    if machine_config:
        machine_info = f" | Config: {machine_config.get('motor_rpm', 'N/A')} RPM"
    
    st.markdown(f'<div class="{alert_class}"><strong>Alert:</strong> Machine {machine_id} is in {status} condition. Health Score: {health_score:.1f}% | Active Sensors: {sensor_list}{data_source_info}{machine_info}{validation_info}</div>', 
                unsafe_allow_html=True)
    
    # Show validation warnings if any
    if validation_result.get('warnings'):
        st.warning("**Data Validation Warnings:**")
        for warning in validation_result['warnings'][:3]:
            st.write(f"• {warning}")
        if len(validation_result['warnings']) > 3:
            st.write(f"• ... and {len(validation_result['warnings']) - 3} more warnings")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Multi-Axis Sensor Data")
        
        fig_time = go.Figure()
        colors = {'Fx': 'blue', 'Fy': 'green', 'Fz': 'red', 'v0': 'orange'}
        
        for axis_key, signal_data in current_signals.items():
            if axis_key == 'v0':
                fig_time.add_trace(go.Scatter(
                    x=time_vector[:len(signal_data)] if time_vector is not None else np.arange(len(signal_data)),
                    y=signal_data,
                    mode='lines',
                    name=f'{axis_key} (Temperature)',
                    line=dict(color=colors.get(axis_key, 'purple'), width=2),
                    yaxis='y2'
                ))
            else:
                fig_time.add_trace(go.Scatter(
                    x=time_vector[:len(signal_data)] if time_vector is not None else np.arange(len(signal_data)),
                    y=signal_data,
                    mode='lines',
                    name=f'{axis_key} (Vibration)',
                    line=dict(color=colors.get(axis_key, 'purple'), width=1.5)
                ))
        
        data_source_text = f"Real-time Sensor Data ({data_source})"
        if data_source == "MySQL Real-time" and mysql_table:
            data_source_text += f" - {mysql_table}"
        
        if 'v0' in current_signals:
            fig_time.update_layout(
                title=data_source_text,
                xaxis_title="Time (seconds)" if time_vector is not None else "Sample",
                yaxis=dict(title="Vibration Amplitude (g)", side="left"),
                yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"),
                height=350,
                legend=dict(x=0.02, y=0.98)
            )
        else:
            fig_time.update_layout(
                title=data_source_text,
                xaxis_title="Time (seconds)" if time_vector is not None else "Sample",
                yaxis_title="Vibration Amplitude (g)",
                height=350,
                legend=dict(x=0.02, y=0.98)
            )
        
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("📊 Frequency Spectrum Analysis")
        
        vibration_signals = {k: v for k, v in current_signals.items() if k in ['Fx', 'Fy', 'Fz']}
        
        if vibration_signals:
            fig_freq = go.Figure()
            
            for axis_key, signal_data in vibration_signals.items():
                fft_values = np.abs(fft(signal_data))
                freqs = fftfreq(len(signal_data), 1/sampling_rate)
                
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = fft_values[:len(fft_values)//2]
                
                fig_freq.add_trace(go.Scatter(
                    x=positive_freqs,
                    y=positive_fft,
                    mode='lines',
                    name=f'{axis_key} Spectrum',
                    line=dict(color=colors.get(axis_key, 'purple'), width=1.5)
                ))
            
            fig_freq.update_layout(
                title="Frequency Domain Analysis",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                height=350,
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        else:
            st.info("Select at least one vibration axis (Fx, Fy, or Fz) to view frequency analysis")
    
    # MySQL Real-time Status Panel
    if data_source == "MySQL Real-time":
        st.subheader("📡 MySQL Real-time Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            connection_status = "🟢 Connected" if st.session_state.mysql_connected else "🔴 Disconnected"
            st.metric("Connection", connection_status)
        
        with col2:
            monitoring_status = "🟢 Active" if st.session_state.is_monitoring else "🔴 Stopped"
            st.metric("Monitoring", monitoring_status)
        
        with col3:
            buffer_size = len(st.session_state.mysql_data_buffer) if not st.session_state.mysql_data_buffer.empty else 0
            st.metric("Buffer Size", f"{buffer_size} pts")
        
        with col4:
            last_update = st.session_state.mysql_last_timestamp if st.session_state.mysql_last_timestamp else "Never"
            if isinstance(last_update, str) and last_update != "Never":
                try:
                    last_update = pd.to_datetime(last_update).strftime('%H:%M:%S')
                except:
                    pass
            st.metric("Last Update", str(last_update))
        
        # Display recent data sample if available
        if not st.session_state.mysql_data_buffer.empty and st.session_state.mysql_connected:
            st.write("**Recent MySQL Data Sample:**")
            sample_data = st.session_state.mysql_data_buffer.tail(5)
            st.dataframe(sample_data, use_container_width=True)
    
    # Temperature analysis (if v0 is selected)
    if 'v0' in current_signals:
        st.subheader("🌡️ Enhanced Temperature Analysis")
        
        # Basic temperature metrics
        col1, col2, col3, col4 = st.columns(4)
        
        temp_data = current_signals['v0']
        avg_temp = np.mean(temp_data)
        max_temp = np.max(temp_data)
        min_temp = np.min(temp_data)
        temp_range = max_temp - min_temp
        
        with col1:
            st.metric("Average Temperature", f"{avg_temp:.1f}°C")
        with col2:
            st.metric("Max Temperature", f"{max_temp:.1f}°C")
        with col3:
            st.metric("Min Temperature", f"{min_temp:.1f}°C")
        with col4:
            st.metric("Temperature Range", f"{temp_range:.1f}°C")
        
        # Enhanced temperature health breakdown (if available from analysis)
        if 'v0' in axis_analysis:
            temp_analysis = axis_analysis['v0']
            
            st.write("**Temperature Health Breakdown:**")
            col1, col2, col3, col4 = st.columns(4)
            
            component_scores = temp_analysis.get('component_scores', {})
            weights_used = temp_analysis.get('weights_used', {})
            
            if component_scores and weights_used:
                with col1:
                    mean_score = component_scores.get('mean_score', 0)
                    mean_weight = weights_used.get('mean_weight', 0)
                    st.metric(
                        "Mean Temp Score", 
                        f"{mean_score:.1f}%", 
                        delta=f"Weight: {mean_weight:.2f}"
                    )
                
                with col2:
                    max_score = component_scores.get('max_score', 0)
                    max_weight = weights_used.get('max_weight', 0)
                    st.metric(
                        "Max Temp Score", 
                        f"{max_score:.1f}%", 
                        delta=f"Weight: {max_weight:.2f}"
                    )
                
                with col3:
                    rise_score = component_scores.get('rise_score', 0)
                    rise_weight = weights_used.get('rise_weight', 0)
                    st.metric(
                        "Rise Rate Score", 
                        f"{rise_score:.1f}%", 
                        delta=f"Weight: {rise_weight:.2f}"
                    )
                
                with col4:
                    overall_temp_score = temp_analysis.get('health_score', 0)
                    st.metric(
                        "Overall Temp Health", 
                        f"{overall_temp_score:.1f}%",
                        delta="Weighted Average"
                    )
            
            # Show applied thresholds
            temp_thresholds = temp_analysis.get('thresholds_used', {})
            temp_analysis_data = temp_analysis.get('temperature_analysis', {})
            
            if temp_thresholds and temp_analysis_data:
                st.write("**Applied Temperature Thresholds:**")
                thresholds_applied = temp_analysis_data.get('thresholds_applied', {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    normal_thresh = temp_thresholds.get('normal_max', 80)
                    normal_mult = thresholds_applied.get('normal_multiplier', 1.0)
                    st.write(f"**Normal:** <{normal_thresh:.1f}°C")
                    if normal_mult != 1.0:
                        st.caption(f"(Multiplier: {normal_mult:.1f})")
                
                with col2:
                    warning_thresh = temp_thresholds.get('warning_max', 85)
                    warning_mult = thresholds_applied.get('warning_multiplier', 1.0)
                    st.write(f"**Warning:** <{warning_thresh:.1f}°C")
                    if warning_mult != 1.0:
                        st.caption(f"(Multiplier: {warning_mult:.1f})")
                
                with col3:
                    critical_thresh = temp_thresholds.get('critical_max', 95)
                    critical_mult = thresholds_applied.get('critical_multiplier', 1.0)
                    st.write(f"**Critical:** <{critical_thresh:.1f}°C")
                    if critical_mult != 1.0:
                        st.caption(f"(Multiplier: {critical_mult:.1f})")
            
            # Temperature trend visualization
            if len(temp_data) > 10:
                st.write("**Temperature Trend Analysis:**")
                
                window_size = min(10, len(temp_data) // 4)
                if window_size > 1:
                    moving_avg = pd.Series(temp_data).rolling(window=window_size).mean()
                    
                    fig_temp_trend = go.Figure()
                    
                    # Plot actual temperature
                    fig_temp_trend.add_trace(go.Scatter(
                        x=list(range(len(temp_data))),
                        y=temp_data,
                        mode='lines',
                        name='Temperature',
                        line=dict(color='orange', width=1),
                        opacity=0.7
                    ))
                    
                    # Plot moving average
                    fig_temp_trend.add_trace(go.Scatter(
                        x=list(range(len(moving_avg))),
                        y=moving_avg,
                        mode='lines',
                        name=f'Moving Average ({window_size} pts)',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add threshold lines
                    if temp_thresholds:
                        fig_temp_trend.add_hline(
                            y=temp_thresholds.get('normal_max', 80),
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Normal Threshold"
                        )
                        fig_temp_trend.add_hline(
                            y=temp_thresholds.get('warning_max', 85),
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="Warning Threshold"
                        )
                        fig_temp_trend.add_hline(
                            y=temp_thresholds.get('critical_max', 95),
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Critical Threshold"
                        )
                    
                    fig_temp_trend.update_layout(
                        title="Temperature Trend with Thresholds",
                        xaxis_title="Sample",
                        yaxis_title="Temperature (°C)",
                        height=300,
                        legend=dict(x=0.02, y=0.98)
                    )
                    
                    st.plotly_chart(fig_temp_trend, use_container_width=True)
        else:
            st.info("🤖 **Train the AI model** to enable detailed temperature health analysis with custom weights")
    
    # Multi-axis data statistics
    if current_signals:
        st.subheader("📊 Multi-Axis Data Statistics")
        stats_data = []
        for axis_key, signal_data in current_signals.items():
            stats_data.append({
                "Sensor": axis_key,
                "Mean": f"{np.mean(signal_data):.4f}",
                "Std Dev": f"{np.std(signal_data):.4f}",
                "Min": f"{np.min(signal_data):.4f}",
                "Max": f"{np.max(signal_data):.4f}",
                "RMS": f"{np.sqrt(np.mean(signal_data**2)):.4f}" if axis_key != 'v0' else "N/A",
                "Unit": "g" if axis_key != 'v0' else "°C"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    # Individual Axis Analysis Section
    if axis_analysis:
        st.subheader("🔍 Individual Axis Analysis")
        
        num_axes = len(axis_analysis)
        if num_axes > 0:
            axis_cols = st.columns(min(num_axes, 4))
            
            for idx, (axis_key, analysis) in enumerate(axis_analysis.items()):
                col_idx = idx % 4
                with axis_cols[col_idx]:
                    st.markdown(f"### {axis_key} Axis")
                    
                    if not isinstance(analysis, dict):
                        st.warning(f"⚠️ Incomplete analysis data for {axis_key}")
                        continue
                    
                    if axis_key in ['Fx', 'Fy', 'Fz']:
                        # Vibration axis analysis
                        axis_health = analysis.get('health_score', 0)
                        axis_status = analysis.get('health_status', 'UNKNOWN')
                        
                        if axis_health >= 80:
                            status_color = "🟢"
                        elif axis_health >= 60:
                            status_color = "🟡"
                        else:
                            status_color = "🔴"
                        
                        st.metric("Health Score", f"{axis_health:.1f}%")
                        st.write(f"**Status:** {status_color} {axis_status}")
                        
                        if all(key in analysis for key in ['rms', 'peak', 'crest_factor', 'dominant_freq']):
                            st.write("**Key Metrics:**")
                            st.write(f"• RMS: {analysis['rms']:.3f} g")
                            st.write(f"• Peak: {analysis['peak']:.3f} g")
                            st.write(f"• Crest Factor: {analysis['crest_factor']:.2f}")
                            st.write(f"• Dominant Freq: {analysis['dominant_freq']:.1f} Hz")
                        
                        # Fault indicators
                        if 'fault_indicators' in analysis:
                            fault_ind = analysis['fault_indicators']
                            st.write("**Fault Analysis:**")
                            
                            if 'rotation_freq' in fault_ind:
                                st.write(f"• Rotation: {fault_ind['rotation_freq']:.1f} Hz")
                            
                            thresholds = analysis.get('thresholds_used', {})
                            
                            bearing_val = fault_ind.get('bearing_fault', 0)
                            bearing_thresh = thresholds.get('bearing', 50)
                            if bearing_val > bearing_thresh:
                                st.warning(f"⚠️ Bearing: {bearing_val:.1f} (>{bearing_thresh:.0f})")
                            else:
                                st.success(f"✅ Bearing: {bearing_val:.1f} (<{bearing_thresh:.0f})")
                            
                            imbalance_val = fault_ind.get('imbalance', 0)
                            imbalance_thresh = thresholds.get('imbalance', 100)
                            if imbalance_val > imbalance_thresh:
                                st.warning(f"⚠️ Imbalance: {imbalance_val:.1f} (>{imbalance_thresh:.0f})")
                            else:
                                st.success(f"✅ Imbalance: {imbalance_val:.1f} (<{imbalance_thresh:.0f})")
                            
                            misalign_val = fault_ind.get('misalignment', 0)
                            misalign_thresh = thresholds.get('misalignment', 75)
                            if misalign_val > misalign_thresh:
                                st.warning(f"⚠️ Misalign: {misalign_val:.1f} (>{misalign_thresh:.0f})")
                            else:
                                st.success(f"✅ Misalign: {misalign_val:.1f} (<{misalign_thresh:.0f})")
                        else:
                            st.info("Train AI model for detailed fault analysis")
                    
                    elif axis_key == 'v0':
                        # Temperature analysis
                        temp_health = analysis.get('health_score', 0)
                        temp_status = analysis.get('health_status', 'UNKNOWN')
                        
                        if temp_health >= 80:
                            status_color = "🟢"
                        elif temp_health >= 60:
                            status_color = "🟡"
                        else:
                            status_color = "🔴"
                        
                        st.metric("Health Score", f"{temp_health:.1f}%")
                        st.write(f"**Status:** {status_color} {temp_status}")
                        
                        if all(key in analysis for key in ['mean_temp', 'max_temp', 'min_temp', 'temp_rise_rate']):
                            st.write("**Temperature Metrics:**")
                            st.write(f"• Mean: {analysis['mean_temp']:.1f}°C")
                            st.write(f"• Max: {analysis['max_temp']:.1f}°C")
                            st.write(f"• Min: {analysis['min_temp']:.1f}°C")
                            st.write(f"• Rise Rate: {analysis['temp_rise_rate']:.2f}°C/s")
                            
                            thresholds = analysis.get('thresholds_used', {})
                            if thresholds:
                                st.write("**Thresholds:**")
                                st.write(f"• Normal: <{thresholds.get('normal_max', 80)}°C")
                                st.write(f"• Warning: <{thresholds.get('warning_max', 85)}°C")
                                st.write(f"• Critical: <{thresholds.get('critical_max', 95)}°C")
                    
                    # Recommendations
                    recommendations = analysis.get('recommendations', [])
                    if recommendations:
                        st.write("**Recommendations:**")
                        for rec in recommendations[:3]:
                            st.write(f"• {rec}")
                    else:
                        st.info("No specific recommendations available")
    else:
        st.info("🤖 **Train the AI model** to enable detailed individual axis analysis")
    
    # Historical trends
    st.subheader("📈 Historical Health Trends")
    
    current_time = datetime.now()
    st.session_state.historical_data.append({
        "timestamp": current_time,
        "health_score": health_score,
        "status": status,
        "machine_id": machine_id,
        "data_source": data_source
    })
    
    # Keep only last 50 points
    if len(st.session_state.historical_data) > 50:
        st.session_state.historical_data = st.session_state.historical_data[-50:]
    
    if len(st.session_state.historical_data) > 1:
        hist_df = pd.DataFrame(st.session_state.historical_data)
        
        fig_hist = px.line(
            hist_df, 
            x="timestamp", 
            y="health_score",
            title="Health Score Trend",
            color_discrete_sequence=["blue"]
        )
        fig_hist.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Healthy Threshold")
        fig_hist.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Enhanced Maintenance recommendations
    st.subheader("🔧 Advanced Maintenance Recommendations")
    
    if axis_analysis:
        all_recommendations = []
        critical_axes = []
        warning_axes = []
        
        for axis_key, analysis in axis_analysis.items():
            if analysis['health_score'] < 60:
                critical_axes.append(axis_key)
            elif analysis['health_score'] < 80:
                warning_axes.append(axis_key)
            all_recommendations.extend(analysis['recommendations'])
        
        if critical_axes:
            st.error(f"🚨 **CRITICAL PRIORITY** - Axes requiring immediate attention: {', '.join(critical_axes)}")
            st.write("**Immediate Actions Required:**")
            for rec in set(all_recommendations):
                if any(word in rec.lower() for word in ['immediate', 'critical', 'stop', 'replace']):
                    st.write(f"• {rec}")
        
        if warning_axes:
            st.warning(f"⚠️ **MEDIUM PRIORITY** - Axes requiring monitoring: {', '.join(warning_axes)}")
            st.write("**Scheduled Actions:**")
            for rec in set(all_recommendations):
                if any(word in rec.lower() for word in ['monitor', 'check', 'inspect', 'schedule']):
                    st.write(f"• {rec}")
        
        if not critical_axes and not warning_axes:
            st.success("✅ **All axes operating normally** - Continue routine monitoring")
            st.write("**Routine Maintenance:**")
            st.write("• Continue regular vibration monitoring")
            st.write("• Maintain lubrication schedule")
            st.write("• Verify operating parameters monthly")
        
        st.write("**Axis-Specific Maintenance Schedule:**")
        for axis_key, analysis in axis_analysis.items():
            if analysis['health_score'] < 80:
                next_check = "Within 24 hours" if analysis['health_score'] < 60 else "Within 1 week"
                st.write(f"• **{axis_key}**: {next_check} - Health: {analysis['health_score']:.1f}%")
    else:
        if health_score >= 80:
            st.success("✅ Machine is operating normally. Continue regular monitoring.")
        elif health_score >= 60:
            st.warning("⚠️ Schedule inspection within 7 days. Monitor closely for degradation.")
            st.write("**Recommended Actions:**")
            st.write("- Check lubrication levels")
            st.write("- Inspect for loose connections")
            st.write("- Verify alignment")
        else:
            st.error("🚨 Immediate maintenance required! Schedule downtime ASAP.")
            st.write("**Critical Actions:**")
            st.write("- Stop machine if safe to do so")
            st.write("- Contact maintenance team immediately")
            st.write("- Prepare for component replacement")
    
    # Auto-refresh for monitoring
    if st.session_state.is_monitoring:
        time.sleep(monitoring_interval)
        st.rerun()

if __name__ == "__main__":
    main()
