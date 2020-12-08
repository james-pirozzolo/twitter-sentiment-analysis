const form = document.getElementById('sentiment-form');
const API_URL = window.location.href

form.addEventListener('submit', (event) => {
    event.preventDefault();
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
      alert(score);
    })
});