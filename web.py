from flask import Flask, render_template_string, request

app = Flask(__name__)

@app.route('/')
def index():
    # HTML template with a button that triggers the trigger() function
    html_template = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Button Trigger</title>
      </head>
      <body>
        <div style="text-align: center; margin-top: 50px;">
          <h1>Press the button to trigger the function</h1>
          <form action="/trigger" method="post">
            <button type="submit">Press me</button>
          </form>
        </div>
      </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/spot', methods=['POST'])
def trigger():
    # Function to be triggered by the button press
    print("Button was pressed!")
    return "Button was pressed!"

if __name__ == '__main__':
    app.run(debug=True)
