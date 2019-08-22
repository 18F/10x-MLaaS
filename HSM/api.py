from flask import Flask, jsonify, make_response, send_file
from flask_httpauth import HTTPBasicAuth
from utils import config, db_utils
from werkzeug.security import check_password_hash
from prediction_manager import PredictionManager
from machine_learning_manager import MachineLearningManager
from data_manager import QualtricDataManager

# initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = config.APP_SECRET_KEY
auth = HTTPBasicAuth()
prediction_manager = PredictionManager(QualtricDataManager(),
                                       MachineLearningManager(None, 'model_sw.pkl', 's3'))


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


# This will download the data and predict the new batch of data from survey
@app.route('/predict', methods=['GET'])
@auth.login_required
def predict():
    results_path, df, id_pred_map, outfile = prediction_manager.predict()
    # df = prediction_manager.predict()
    # model = prediction_manager.ml_mgr.get_model()
    print("done with prediction")
    print(results_path)
    # path = os.path.join(results_path, outfile)
    # path = os.path.join('/home/hsm', 'model', 'results', 'ClassificationResults.xlsx')
    # with open(path, 'rb') as f:
    return send_file(open(results_path, 'rb'),
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                     as_attachment=True,
                     attachment_filename=outfile)
    # with io.BytesIO() as xlsx_file:
    #     writer = pd.ExcelWriter(output, engine='xlsxwriter')
    # def generate():
    #     for row in iter_all_rows():
    #         yield ','.join(row) + '\n'
    # return jsonify({'task': 'predict',
    #                 'username': auth.username()
    #                 })


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
    # db.dal.connect()
    # session = db.dal.Session()
    port = int(config.APP_PORT)
    app.run(host='0.0.0.0', port=port)
