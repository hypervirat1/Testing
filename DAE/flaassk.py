from flask import Flask,render_template,request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
@app.route('/')
def xyz():
    return render_template("xyz.html")
@app.route("/team_members")
def team_members():
    return render_template("team_members.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route('/Trinity')
def Trinity():
    return render_template('Trinity.html')
@app.route('/about_inside')
def about_inside():
    return render_template('about_inside.html')
@app.route('/handd')
def handd():
    return render_template('handd.html')

if __name__ == '__main__':
    app.run()



