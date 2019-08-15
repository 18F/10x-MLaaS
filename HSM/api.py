from flask import Flask, jsonify, make_response
from flask_httpauth import HTTPBasicAuth
from utils import config, db, db_utils
from werkzeug.security import check_password_hash

# initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = config.APP_SECRET_KEY
auth = HTTPBasicAuth()


@auth.verify_password
def verify_password(username, password):
    if username in config.users:
        return check_password_hash(config.users.get(username), password)
    return False


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access'}), 401)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/')
@auth.login_required
def index():
    return jsonify({'title': 'Predict->Validate->Train',
                    'username': auth.username()
                    })


@app.route('/predict', methods=['GET'])
@auth.login_required
def predict():
    return jsonify({'task': 'predict',
                    'username': auth.username()
                    })


@app.route('/validate')  # , methods=['POST'])
@auth.login_required
def validate():
    return jsonify({'task': 'validate',
                    'username': auth.username(),
                    }), 200


@app.route('/train')  # , methods=['POST'])
@auth.login_required
def train():

    return jsonify({'task': 'train',
                    'username': auth.username()
                    }), 200


if __name__ == "__main__":
    db_utils.create_postgres_db()
    db.dal.connect()
    session = db.dal.Session()
    port = int(config.APP_PORT)
    app.run(host='0.0.0.0', port=port)
