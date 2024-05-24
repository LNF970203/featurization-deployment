from flask import Flask, request, jsonify
import logging

from Utils import response_utils as rutils
from Utils import image_utils as iutils
from Utils import model_utils as mutils


app = Flask(__name__)

# Configure logging
# This is best best prectices for debugging
logging.basicConfig(level=logging.DEBUG)

@app.route("/inference", methods = ["POST"])
def inference():
    if request.method == "POST":
        data = request.json
        params = data["body"]
        image_data = params["image"]

        # load the row image
        try:
            payload = iutils.get_image_payload(image_data)
            app.logger.info(payload)
            process_payload = iutils.preprocess_image_input(payload)
            app.logger.info(process_payload)

            # load the model
            selected_model = mutils.load_featurized_model()
            predictions = selected_model.predict(process_payload)
            app.logger.info(predictions)

            # send the response
            return jsonify(rutils.success_response({"id": 12}))

        except Exception as error:
            app.logger.info(str(error))
            message = "unable to get features"
            error = {
                "errorCode": None,
                "message": message,
                "suggestions": None
            }
            return jsonify(rutils.failuer_response([error]))
        

if __name__ == "__main__":
    app.run(debug=True)