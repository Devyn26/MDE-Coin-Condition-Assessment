from flask import Flask, render_template
from forms import WebForm


app = Flask(__name__)


def run():

    app.run(host='localhost', port=4000)
    home()


@app.route('/', methods=['GET', 'POST'])
def home():
    form = WebForm

    return render_template('web.html', form=form)


if __name__ == "__main__":
    run()
