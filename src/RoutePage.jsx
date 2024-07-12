import SignIn from './SignIn.jsx';
import SignUp from './SignUp.jsx';
import {BrowserRouter as Router, Routes, Route, } from "react-router-dom";
import TermsAndConditons from './TermsAndConditions.jsx';
import Home from './Home.jsx';


const RoutePage = () => {
    return (  
      <>
        {/* //Routing to link pages */}
      <Router>
         <Routes>
          <Route exact path="/" element={<SignIn/>}/>
          <Route exact path="/SignUp.jsx" element={<SignUp/>}/>
         <Route exact path="/TermsAndConditions.jsx" element={<TermsAndConditons />} />
         <Route exact path="/Home.jsx" element={<Home />} />
        </Routes>
        </Router>
        </>
    );
}
 
export default RoutePage;