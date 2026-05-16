readline = require 'readline'

rl = readline.createInterface
  input: process.stdin
  output: process.stdout


enter -> console.log("please run directly")
rl.question "Enter your name: ", (answer) ->
console.log "Thank you for Testing my AI, #{answer}! Please give feedback on the github page specified in package.json"
rl.close()
