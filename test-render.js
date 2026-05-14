const http = require('https');
const options = {
  hostname: 'plant-leaf-desease.onrender.com',
  port: 443,
  path: '/extract-leaves',
  method: 'POST'
};
const req = http.request(options, res => {
  console.log(`statusCode: ${res.statusCode}`);
  res.on('data', d => {
    process.stdout.write(d);
  });
});
req.on('error', error => {
  console.error(error);
});
req.end();
