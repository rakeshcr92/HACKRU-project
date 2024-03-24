from flask import Flask







def create_app():
    app = Flask(__name__)
    

    from .views import views
    from .prediction import prediction
    
    

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(prediction, url_prefix='/')
    

    
  

    



    return app




