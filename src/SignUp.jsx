import fns_extract from './images/fns_extract.png'
import FNS from './images/FNS.png'
import './index.css';
import {Link} from "react-router-dom";
import {auth} from './firebase.jsx'
import { createUserWithEmailAndPassword } from 'firebase/auth';
import {useState} from 'react'
import {useNavigate} from 'react-router-dom'


// Passing props from parent app.js
const SignUp = (/*name,email,password,InputName, InputEmail, InputPassword*/) => {
  const [name,setName]= useState('')
  const [email,setEmail]= useState('')
  const [password,setPassword]= useState('')
  const move = useNavigate()

  const HandleSubmit = async(e)=> { 
    e.preventDefault()
    try{
        await createUserWithEmailAndPassword(auth,email,password)
        console.log("account created")
        move("/Home.jsx")
    } catch(error){
        console.log(error)
    }
  }

    

    return ( 
       
            <div className="signCard" style={{backgroundImage: 'url(' + FNS + ')'}}>

                <img id="logo" src={fns_extract} alt="logo"/>
                
        
            <form id='signup-form' >
                 <p className="Welcome">Get Started </p>
                <fieldset>
                    <label htmlFor="username"> 
                        Username:
                        <input type="text" 
                             id="username"
                             required
                            //  setting value to user input as input is made
                            value={name}
                            onChange={(e)=>setName(e.target.value)}
                     />
                    </label>
                    <label htmlFor="email-address"> 
                        Email Address:
                        <input type="email"
                        id="email-address" 
                         required 
                        value={email}
                        onChange={(e)=> setEmail(e.target.value)}
                        />
                    </label>
                    <label htmlFor="passKey">
                        Password:
                        <input type="password" 
                        id="passKey"
                        required
                        value={password}
                        onChange={(e)=> setPassword(e.target.value)}
                        />
                    </label>
                    
                </fieldset>
            
                <input className="terms" type="radio" name="terms-conditions" value="true"/><span>I agree to to the</span>
            <Link className='terms-conditions' to="/TermsAndConditions.jsx">Terms and Conditions</Link>
            

            <fieldset>
               
                <button type="submit" className="submit-button" onSubmit={HandleSubmit} >Sign Up</button>
                
                <p>Have an account already?<Link className='signIn' to="/">SignIn</Link></p>
            </fieldset>

            </form>
        </div>
   
     );
}
 
export default SignUp;

// const provider = new GoogleAuthProvider(); // include other third-party providers
// // Open a popup window or redirect to a sign-in page
// signInWithPopup (auth, provider)
// .then((result) => {
// // The signed-in user info.
// const user = result.user;
// //
// }).catch((error) => {
// // Handle Errors here.
// const errorCode = error.code;
// const errorMessage = error.message;
// //
// });
