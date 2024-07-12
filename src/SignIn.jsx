import './index.css';
import fns_extract from './images/fns_extract.png'
import FNS from './images/FNS.png'
import {Link} from "react-router-dom";
import {auth,} from './firebase.jsx'
import {GoogleAuthProvider,signInWithPopup } from 'firebase/auth';
import {useState} from 'react'
import { signInWithEmailAndPassword } from 'firebase/auth';



const SignIn = () => {
    const [email,setEmail]= useState('')
    const [password,setPassword]= useState('')

   
  
    const HandleSubmit = async(e)=> { 
      e.preventDefault()
      try{
          await signInWithEmailAndPassword(auth,email,password)
          console.log("login successful")
          move("/Home.jsx")
      } catch(err){
          console.log(err.message)
          console.log(err.code)
      }
    }

    const GoogleSignIn = async(e)=> { 
        e.preventDefault()
        try{
            await signInWithPopup(auth,GoogleAuthProvider)
            console.log("login successful")
            move("/Home.jsx")
        } catch(err){
            console.log(err.message)
            console.log(err.code)
        }
      }
  
  


    return (

     <div className="signCard" style={{backgroundImage: 'url(' + FNS + ')'}}>

         <img id="logo" src={fns_extract} alt="logo"/> 
        
        
        <form id="signin-form">
            <p className="Welcome">Welcome back! </p>
            <fieldset>
                <label htmlFor="email-address"> 
                    Email Address:
                    <input 
                    type="email" 
                    id="email-address"        
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    />
                </label>
                {/* {console.log(e.target.value)} */}
                <label htmlFor="passKey">Password:
                    <input type="password"
                     id="passKey"
                      required
                      value={password}
                      onChange={(e)=> setPassword(e.target.value)}
                    />
                      <p><a href="">Forgot password?</a></p>
                </label>
                
            </fieldset>

            <fieldset>
                
                <button type="submit" className="submit-button" onSubmit={HandleSubmit}>Sign In</button>
                <button type="submit" className="submit-button" onSubmit={GoogleSignIn}>Sign in with Google</button>
                <p>Don't have an account?<Link className='signUp' to="/SignUp.jsx">SignUp</Link></p>
            </fieldset>

        </form>
    </div>
  
      );
}
 
export default SignIn;
