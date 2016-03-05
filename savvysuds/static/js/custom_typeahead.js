//typeahead
var beer_list = new Bloodhound({
  // the JSON key to search through
  datumTokenizer: Bloodhound.tokenizers.obj.whitespace('beer_info'),
  // tokenizes the user input (I think)
  queryTokenizer: Bloodhound.tokenizers.whitespace,
  // tells bloodhound where the data lives (might use prefetch for commonly used stuff)
  // remote: {
  //   url: "{{ url_for('static', filename='assets/beers.json') }}",
  // }
  prefetch: {
  url: '{{ url_for('static', filename='assets/beers.json') }}'
  }
});

$('#user-beer1').typeahead({
  hint: true,
  highlight: true,
  minLength: 1
},
{
  name: 'beer_list',
  // JSON key of thing to display
  displayKey: 'beer_info', // can be a function - return gets displayed
  source: beer_list
}).bind('typeahead:selected', function(obj, datum, name) {
  var idField = $('#user-beer1-id');
  idField.val(datum.id);
});

$('#user-beer2').typeahead({
  hint: true,
  highlight: true,
  minLength: 1
},
{
  name: 'beer_list',
  // JSON key of thing to display
  displayKey: 'beer_info', // can be a function - return gets displayed
  source: beer_list
}).bind('typeahead:selected', function(obj, datum, name) {
  var idField = $('#user-beer2-id');
  idField.val(datum.id);
});

$('#user-beer3').typeahead({
  hint: true,
  highlight: true,
  minLength: 1
},
{
  name: 'beer_list',
  // JSON key of thing to display
  displayKey: 'beer_info', // can be a function - return gets displayed
  source: beer_list
}).bind('typeahead:selected', function(obj, datum, name) {
  var idField = $('#user-beer3-id');
  idField.val(datum.id);
});
