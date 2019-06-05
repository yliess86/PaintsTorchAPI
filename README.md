![LOGO](res/logo.png)

API for PaintsTorch, an automatic colorizer.
The model has been train on Anime style images only.

## I. API
Json data is required.
The request and response format are the following.

### II.a. Request
```json
{
  "sketch"  : "BASE64 PNG/JPG/JPEG",
  "hint"    : "BASE64 PNG With Transparence",
  "opcaity" : 0.0 /* Opcaity can vary from 0 to 1 */
}
```

### II.b. Response
```json
{
  "colored"  : "BASE64 PNG"
}
```

## II. Example
Here is an example of how to query the API using Javascript with ajax.
```js
let data    = { 'sketch': sketch, 'hint': hint, 'opacity': opacity };
data        = JSON.stringify(data);

$.ajax({
  url         : 'https://dvic.devinci.fr/dgx/paints_torch/api/v1/colorizer',
  type        : 'POST',
  data        : data,
  contentType : 'application/json; charset=utf-8',
  dataType    :'json',
  success     : function(response){
    if('colored' in response) {
      let colored = response.color;
      // Do wathever you want with it
    }
  }
})
```
