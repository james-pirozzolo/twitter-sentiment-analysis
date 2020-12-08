const form = document.getElementById('sentiment-form');
const spinner = document.getElementById('sentiment-loader');

const API_URL = window.location.href;

// bar to be used to show sentiment 
var bar = new ProgressBar.Line(container, {
  strokeWidth: 4,
  easing: 'easeInOut',
  duration: 1400,
  color: '#FFEA82',
  trailColor: '#eee',
  trailWidth: 1,
  svgStyle: {width: '100%', height: '100%'},
  // to transition style
  from: {color: '#ED6A5A'},
  to: {color: '#2BD469'},
  step: (state, bar) => {
    bar.path.setAttribute('stroke', state.color);
    bar.setText(Math.round(bar.value()*100)+ ' %')
  },

});

form.addEventListener('submit', (event) => {
    event.preventDefault();
    spinner.classList.remove('collapse');
    console.log('form successfully submitted');
    const formData = new FormData(form);
    const rawTweet = formData.get('tweet').trim();

    const tweet = { rawTweet };
    fetch(`${API_URL}tweet_sentiment`, {
        method: 'POST',
        body: JSON.stringify(tweet),
        headers: {
            'content-type': 'application/json'
        }
    }).then(res => res.json())
    .then(data => {
      score = data.neg_sentiment;
      spinner.classList.add('collapse');
      // alert(score);
      score === -1 ? console.log('error retrieving tweet') : bar.animate(1-score);
    })
});