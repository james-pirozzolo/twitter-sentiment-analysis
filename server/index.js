const path = require('path');
// require('dotenv').config({path: path.join(__dirname, '../.env')});
const express = require('express');
const http = require('http');
const bodyParser = require('body-parser');
// const cors = require('cors');

//instantiate server
const app = express();
const port = process.env.PORT || 8080;
app.set('port', port);
const server = http.createServer(app);

// middleware
app.use(express.static(__dirname + '/public'));
app.use(bodyParser.urlencoded({
    extended: true
}));
app.use(bodyParser.json());

// set ejs as view engine
app.set('views', path.join(__dirname, '/views'));
app.set('view engine', 'ejs');

// Starts the server.
server.listen(port, function () {
    console.log(`Starting server on port ${port}`);
});

// home page
app.get('/', (req, res) => {
    res.render('pages/home_page');
});

// sentiment score
app.post('/tweet_sentiment', (req, res) => {
    rawTweet = req.body.rawTweet;
    console.log(rawTweet);
    runPy(rawTweet).then((fromRunpy) => {
        neg_sentiment = parseFloat(fromRunpy.toString().trim());
        console.log(neg_sentiment);
        res.json({
            neg_sentiment
        });
        res.end();
    }).catch(() => {        
        res.json({
            'neg_sentiment': -1.0
        });
        res.end();
    })
})

const runPy = (rawTweet) => {
    return new Promise(function(success, nosuccess) {
        const { spawn } = require('child_process');
        const python_prog = spawn('python', ['../model/repl.py', rawTweet, '../saved_model/'])

        python_prog.stdout.on('data', function(data) {
            success(data);
        });
    });
}


