from flask_wtf import FlaskForm
from wtforms import SubmitField


class WebForm(FlaskForm):
    calculate = SubmitField(label='Calculate')
