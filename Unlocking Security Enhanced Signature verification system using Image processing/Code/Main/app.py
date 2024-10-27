from flask import Flask, render_template, request, redirect, url_for, session
from main import result
# import bcrypt
import threading
import sqlite3
from flask_socketio import SocketIO


app = Flask(__name__,template_folder="template")
socketio = SocketIO(app)
app.secret_key = 'secretKey@123'


# Function to create database and table
def create_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT, password TEXT)''')
    conn.commit()
    conn.close()
    
# Function to handle logout
def logout():
    session.pop('logged_in', None)  # Remove 'logged_in' session variable
    return redirect(url_for('login'))  # Redirect to login page after logout


# def hash_password(password):
#     # Generate a salt and hash the password
#     salt = bcrypt.gensalt()
#     hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
#     return hashed_password


# Route to display the login page
@app.route('/')
def login():
    return render_template('login.html')

# Route to authenticate the login
@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        session['logged_in'] = True
        return redirect(url_for('index'))
    else:
        error_message = 'Invalid username or password'
        return render_template('login.html',error=error_message)

# Route to display the signup page
@app.route('/signup')
def signup():
    return render_template('signup.html')

# Route to handle the signup form submission
@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    # hashed_password = hash_password(password)
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    existing_user = c.fetchone()
    c.execute("SELECT * FROM users WHERE email=?", (email,))
    existing_email = c.fetchone()
    conn.close()

    if existing_user:
        return render_template('signup.html', error='Username already exists')
    elif existing_email:
        return render_template('signup.html', error='Email already exists')
    elif password != confirm_password:
        return render_template('signup.html', error='Passwords do not match')
    else:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))

# Route to display the index page (after successful login)
@app.route('/index', methods=['GET','POST'])
@socketio.on('index')
def index():
    if(request.method=='POST'):
        message="Python app is running on background"
        threading.Thread(target=result).start()
        return render_template('index.html'), "Tkinter application started"
    # else:
    #     message = "Python app is closed"
    #     return render_template('result.html', Results=message), render_template('index.html')
    socketio.emit('recognition_started', {'message': 'Recognition process started'})       
    return render_template('index.html')



if __name__ == '__main__':
    create_table()
    app.run(debug=True)
