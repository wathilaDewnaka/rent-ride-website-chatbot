from flask import Flask

app = Flask(__name__)

# Home route
@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True)
