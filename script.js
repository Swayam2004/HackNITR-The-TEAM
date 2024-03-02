document.getElementById('signInForm').addEventListener('submit', signIn);
document.getElementById('signUpForm').addEventListener('submit', signUp);

function signIn(event) {
  event.preventDefault();
  
  var signInEmail = document.getElementById('signInEmail').value;
  var signInPassword = document.getElementById('signInPassword').value;
  
  // Add your sign-in logic here
  console.log('Sign in with:', signInEmail, signInPassword);
}

function signUp(event) {
  event.preventDefault();
  
  var signUpEmail = document.getElementById('signUpEmail').value;
  var signUpPassword = document.getElementById('signUpPassword').value;
  
  // Add your sign-up logic here
  console.log('Sign up with:', signUpEmail, signUpPassword);
}
