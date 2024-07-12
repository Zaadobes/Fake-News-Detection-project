function nm(e,t){for(var n=0;n<t.length;n++){const r=t[n];if(typeof r!="string"&&!Array.isArray(r)){for(const i in r)if(i!=="default"&&!(i in e)){const o=Object.getOwnPropertyDescriptor(r,i);o&&Object.defineProperty(e,i,o.get?o:{enumerable:!0,get:()=>r[i]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const o of i)if(o.type==="childList")for(const s of o.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&r(s)}).observe(document,{childList:!0,subtree:!0});function n(i){const o={};return i.integrity&&(o.integrity=i.integrity),i.referrerPolicy&&(o.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?o.credentials="include":i.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function r(i){if(i.ep)return;i.ep=!0;const o=n(i);fetch(i.href,o)}})();function rm(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,"default")?e.default:e}var ud={exports:{}},Po={},cd={exports:{}},D={};/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Yr=Symbol.for("react.element"),im=Symbol.for("react.portal"),om=Symbol.for("react.fragment"),sm=Symbol.for("react.strict_mode"),am=Symbol.for("react.profiler"),lm=Symbol.for("react.provider"),um=Symbol.for("react.context"),cm=Symbol.for("react.forward_ref"),dm=Symbol.for("react.suspense"),fm=Symbol.for("react.memo"),hm=Symbol.for("react.lazy"),au=Symbol.iterator;function pm(e){return e===null||typeof e!="object"?null:(e=au&&e[au]||e["@@iterator"],typeof e=="function"?e:null)}var dd={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},fd=Object.assign,hd={};function Jn(e,t,n){this.props=e,this.context=t,this.refs=hd,this.updater=n||dd}Jn.prototype.isReactComponent={};Jn.prototype.setState=function(e,t){if(typeof e!="object"&&typeof e!="function"&&e!=null)throw Error("setState(...): takes an object of state variables to update or a function which returns an object of state variables.");this.updater.enqueueSetState(this,e,t,"setState")};Jn.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,"forceUpdate")};function pd(){}pd.prototype=Jn.prototype;function Ha(e,t,n){this.props=e,this.context=t,this.refs=hd,this.updater=n||dd}var Wa=Ha.prototype=new pd;Wa.constructor=Ha;fd(Wa,Jn.prototype);Wa.isPureReactComponent=!0;var lu=Array.isArray,md=Object.prototype.hasOwnProperty,Ka={current:null},gd={key:!0,ref:!0,__self:!0,__source:!0};function vd(e,t,n){var r,i={},o=null,s=null;if(t!=null)for(r in t.ref!==void 0&&(s=t.ref),t.key!==void 0&&(o=""+t.key),t)md.call(t,r)&&!gd.hasOwnProperty(r)&&(i[r]=t[r]);var a=arguments.length-2;if(a===1)i.children=n;else if(1<a){for(var l=Array(a),u=0;u<a;u++)l[u]=arguments[u+2];i.children=l}if(e&&e.defaultProps)for(r in a=e.defaultProps,a)i[r]===void 0&&(i[r]=a[r]);return{$$typeof:Yr,type:e,key:o,ref:s,props:i,_owner:Ka.current}}function mm(e,t){return{$$typeof:Yr,type:e.type,key:t,ref:e.ref,props:e.props,_owner:e._owner}}function Ga(e){return typeof e=="object"&&e!==null&&e.$$typeof===Yr}function gm(e){var t={"=":"=0",":":"=2"};return"$"+e.replace(/[=:]/g,function(n){return t[n]})}var uu=/\/+/g;function ts(e,t){return typeof e=="object"&&e!==null&&e.key!=null?gm(""+e.key):t.toString(36)}function Ri(e,t,n,r,i){var o=typeof e;(o==="undefined"||o==="boolean")&&(e=null);var s=!1;if(e===null)s=!0;else switch(o){case"string":case"number":s=!0;break;case"object":switch(e.$$typeof){case Yr:case im:s=!0}}if(s)return s=e,i=i(s),e=r===""?"."+ts(s,0):r,lu(i)?(n="",e!=null&&(n=e.replace(uu,"$&/")+"/"),Ri(i,t,n,"",function(u){return u})):i!=null&&(Ga(i)&&(i=mm(i,n+(!i.key||s&&s.key===i.key?"":(""+i.key).replace(uu,"$&/")+"/")+e)),t.push(i)),1;if(s=0,r=r===""?".":r+":",lu(e))for(var a=0;a<e.length;a++){o=e[a];var l=r+ts(o,a);s+=Ri(o,t,n,l,i)}else if(l=pm(e),typeof l=="function")for(e=l.call(e),a=0;!(o=e.next()).done;)o=o.value,l=r+ts(o,a++),s+=Ri(o,t,n,l,i);else if(o==="object")throw t=String(e),Error("Objects are not valid as a React child (found: "+(t==="[object Object]"?"object with keys {"+Object.keys(e).join(", ")+"}":t)+"). If you meant to render a collection of children, use an array instead.");return s}function ci(e,t,n){if(e==null)return e;var r=[],i=0;return Ri(e,r,"","",function(o){return t.call(n,o,i++)}),r}function vm(e){if(e._status===-1){var t=e._result;t=t(),t.then(function(n){(e._status===0||e._status===-1)&&(e._status=1,e._result=n)},function(n){(e._status===0||e._status===-1)&&(e._status=2,e._result=n)}),e._status===-1&&(e._status=0,e._result=t)}if(e._status===1)return e._result.default;throw e._result}var fe={current:null},Ni={transition:null},ym={ReactCurrentDispatcher:fe,ReactCurrentBatchConfig:Ni,ReactCurrentOwner:Ka};function yd(){throw Error("act(...) is not supported in production builds of React.")}D.Children={map:ci,forEach:function(e,t,n){ci(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return ci(e,function(){t++}),t},toArray:function(e){return ci(e,function(t){return t})||[]},only:function(e){if(!Ga(e))throw Error("React.Children.only expected to receive a single React element child.");return e}};D.Component=Jn;D.Fragment=om;D.Profiler=am;D.PureComponent=Ha;D.StrictMode=sm;D.Suspense=dm;D.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=ym;D.act=yd;D.cloneElement=function(e,t,n){if(e==null)throw Error("React.cloneElement(...): The argument must be a React element, but you passed "+e+".");var r=fd({},e.props),i=e.key,o=e.ref,s=e._owner;if(t!=null){if(t.ref!==void 0&&(o=t.ref,s=Ka.current),t.key!==void 0&&(i=""+t.key),e.type&&e.type.defaultProps)var a=e.type.defaultProps;for(l in t)md.call(t,l)&&!gd.hasOwnProperty(l)&&(r[l]=t[l]===void 0&&a!==void 0?a[l]:t[l])}var l=arguments.length-2;if(l===1)r.children=n;else if(1<l){a=Array(l);for(var u=0;u<l;u++)a[u]=arguments[u+2];r.children=a}return{$$typeof:Yr,type:e.type,key:i,ref:o,props:r,_owner:s}};D.createContext=function(e){return e={$$typeof:um,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null,_defaultValue:null,_globalName:null},e.Provider={$$typeof:lm,_context:e},e.Consumer=e};D.createElement=vd;D.createFactory=function(e){var t=vd.bind(null,e);return t.type=e,t};D.createRef=function(){return{current:null}};D.forwardRef=function(e){return{$$typeof:cm,render:e}};D.isValidElement=Ga;D.lazy=function(e){return{$$typeof:hm,_payload:{_status:-1,_result:e},_init:vm}};D.memo=function(e,t){return{$$typeof:fm,type:e,compare:t===void 0?null:t}};D.startTransition=function(e){var t=Ni.transition;Ni.transition={};try{e()}finally{Ni.transition=t}};D.unstable_act=yd;D.useCallback=function(e,t){return fe.current.useCallback(e,t)};D.useContext=function(e){return fe.current.useContext(e)};D.useDebugValue=function(){};D.useDeferredValue=function(e){return fe.current.useDeferredValue(e)};D.useEffect=function(e,t){return fe.current.useEffect(e,t)};D.useId=function(){return fe.current.useId()};D.useImperativeHandle=function(e,t,n){return fe.current.useImperativeHandle(e,t,n)};D.useInsertionEffect=function(e,t){return fe.current.useInsertionEffect(e,t)};D.useLayoutEffect=function(e,t){return fe.current.useLayoutEffect(e,t)};D.useMemo=function(e,t){return fe.current.useMemo(e,t)};D.useReducer=function(e,t,n){return fe.current.useReducer(e,t,n)};D.useRef=function(e){return fe.current.useRef(e)};D.useState=function(e){return fe.current.useState(e)};D.useSyncExternalStore=function(e,t,n){return fe.current.useSyncExternalStore(e,t,n)};D.useTransition=function(){return fe.current.useTransition()};D.version="18.3.1";cd.exports=D;var I=cd.exports;const st=rm(I),wm=nm({__proto__:null,default:st},[I]);/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var _m=I,Sm=Symbol.for("react.element"),Em=Symbol.for("react.fragment"),Im=Object.prototype.hasOwnProperty,km=_m.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,Cm={key:!0,ref:!0,__self:!0,__source:!0};function wd(e,t,n){var r,i={},o=null,s=null;n!==void 0&&(o=""+n),t.key!==void 0&&(o=""+t.key),t.ref!==void 0&&(s=t.ref);for(r in t)Im.call(t,r)&&!Cm.hasOwnProperty(r)&&(i[r]=t[r]);if(e&&e.defaultProps)for(r in t=e.defaultProps,t)i[r]===void 0&&(i[r]=t[r]);return{$$typeof:Sm,type:e,key:o,ref:s,props:i,_owner:km.current}}Po.Fragment=Em;Po.jsx=wd;Po.jsxs=wd;ud.exports=Po;var k=ud.exports,Fs={},_d={exports:{}},Ce={},Sd={exports:{}},Ed={};/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */(function(e){function t(T,A){var x=T.length;T.push(A);e:for(;0<x;){var G=x-1>>>1,Z=T[G];if(0<i(Z,A))T[G]=A,T[x]=Z,x=G;else break e}}function n(T){return T.length===0?null:T[0]}function r(T){if(T.length===0)return null;var A=T[0],x=T.pop();if(x!==A){T[0]=x;e:for(var G=0,Z=T.length,li=Z>>>1;G<li;){var Kt=2*(G+1)-1,es=T[Kt],Gt=Kt+1,ui=T[Gt];if(0>i(es,x))Gt<Z&&0>i(ui,es)?(T[G]=ui,T[Gt]=x,G=Gt):(T[G]=es,T[Kt]=x,G=Kt);else if(Gt<Z&&0>i(ui,x))T[G]=ui,T[Gt]=x,G=Gt;else break e}}return A}function i(T,A){var x=T.sortIndex-A.sortIndex;return x!==0?x:T.id-A.id}if(typeof performance=="object"&&typeof performance.now=="function"){var o=performance;e.unstable_now=function(){return o.now()}}else{var s=Date,a=s.now();e.unstable_now=function(){return s.now()-a}}var l=[],u=[],d=1,f=null,m=3,g=!1,y=!1,w=!1,E=typeof setTimeout=="function"?setTimeout:null,h=typeof clearTimeout=="function"?clearTimeout:null,c=typeof setImmediate<"u"?setImmediate:null;typeof navigator<"u"&&navigator.scheduling!==void 0&&navigator.scheduling.isInputPending!==void 0&&navigator.scheduling.isInputPending.bind(navigator.scheduling);function p(T){for(var A=n(u);A!==null;){if(A.callback===null)r(u);else if(A.startTime<=T)r(u),A.sortIndex=A.expirationTime,t(l,A);else break;A=n(u)}}function v(T){if(w=!1,p(T),!y)if(n(l)!==null)y=!0,Xo(S);else{var A=n(u);A!==null&&Zo(v,A.startTime-T)}}function S(T,A){y=!1,w&&(w=!1,h(N),N=-1),g=!0;var x=m;try{for(p(A),f=n(l);f!==null&&(!(f.expirationTime>A)||T&&!Le());){var G=f.callback;if(typeof G=="function"){f.callback=null,m=f.priorityLevel;var Z=G(f.expirationTime<=A);A=e.unstable_now(),typeof Z=="function"?f.callback=Z:f===n(l)&&r(l),p(A)}else r(l);f=n(l)}if(f!==null)var li=!0;else{var Kt=n(u);Kt!==null&&Zo(v,Kt.startTime-A),li=!1}return li}finally{f=null,m=x,g=!1}}var P=!1,R=null,N=-1,z=5,L=-1;function Le(){return!(e.unstable_now()-L<z)}function nr(){if(R!==null){var T=e.unstable_now();L=T;var A=!0;try{A=R(!0,T)}finally{A?rr():(P=!1,R=null)}}else P=!1}var rr;if(typeof c=="function")rr=function(){c(nr)};else if(typeof MessageChannel<"u"){var su=new MessageChannel,tm=su.port2;su.port1.onmessage=nr,rr=function(){tm.postMessage(null)}}else rr=function(){E(nr,0)};function Xo(T){R=T,P||(P=!0,rr())}function Zo(T,A){N=E(function(){T(e.unstable_now())},A)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(T){T.callback=null},e.unstable_continueExecution=function(){y||g||(y=!0,Xo(S))},e.unstable_forceFrameRate=function(T){0>T||125<T?console.error("forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported"):z=0<T?Math.floor(1e3/T):5},e.unstable_getCurrentPriorityLevel=function(){return m},e.unstable_getFirstCallbackNode=function(){return n(l)},e.unstable_next=function(T){switch(m){case 1:case 2:case 3:var A=3;break;default:A=m}var x=m;m=A;try{return T()}finally{m=x}},e.unstable_pauseExecution=function(){},e.unstable_requestPaint=function(){},e.unstable_runWithPriority=function(T,A){switch(T){case 1:case 2:case 3:case 4:case 5:break;default:T=3}var x=m;m=T;try{return A()}finally{m=x}},e.unstable_scheduleCallback=function(T,A,x){var G=e.unstable_now();switch(typeof x=="object"&&x!==null?(x=x.delay,x=typeof x=="number"&&0<x?G+x:G):x=G,T){case 1:var Z=-1;break;case 2:Z=250;break;case 5:Z=1073741823;break;case 4:Z=1e4;break;default:Z=5e3}return Z=x+Z,T={id:d++,callback:A,priorityLevel:T,startTime:x,expirationTime:Z,sortIndex:-1},x>G?(T.sortIndex=x,t(u,T),n(l)===null&&T===n(u)&&(w?(h(N),N=-1):w=!0,Zo(v,x-G))):(T.sortIndex=Z,t(l,T),y||g||(y=!0,Xo(S))),T},e.unstable_shouldYield=Le,e.unstable_wrapCallback=function(T){var A=m;return function(){var x=m;m=A;try{return T.apply(this,arguments)}finally{m=x}}}})(Ed);Sd.exports=Ed;var Tm=Sd.exports;/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var Pm=I,ke=Tm;function _(e){for(var t="https://reactjs.org/docs/error-decoder.html?invariant="+e,n=1;n<arguments.length;n++)t+="&args[]="+encodeURIComponent(arguments[n]);return"Minified React error #"+e+"; visit "+t+" for the full message or use the non-minified dev environment for full errors and additional helpful warnings."}var Id=new Set,Rr={};function pn(e,t){Hn(e,t),Hn(e+"Capture",t)}function Hn(e,t){for(Rr[e]=t,e=0;e<t.length;e++)Id.add(t[e])}var ut=!(typeof window>"u"||typeof window.document>"u"||typeof window.document.createElement>"u"),zs=Object.prototype.hasOwnProperty,Rm=/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,cu={},du={};function Nm(e){return zs.call(du,e)?!0:zs.call(cu,e)?!1:Rm.test(e)?du[e]=!0:(cu[e]=!0,!1)}function Om(e,t,n,r){if(n!==null&&n.type===0)return!1;switch(typeof t){case"function":case"symbol":return!0;case"boolean":return r?!1:n!==null?!n.acceptsBooleans:(e=e.toLowerCase().slice(0,5),e!=="data-"&&e!=="aria-");default:return!1}}function Am(e,t,n,r){if(t===null||typeof t>"u"||Om(e,t,n,r))return!0;if(r)return!1;if(n!==null)switch(n.type){case 3:return!t;case 4:return t===!1;case 5:return isNaN(t);case 6:return isNaN(t)||1>t}return!1}function he(e,t,n,r,i,o,s){this.acceptsBooleans=t===2||t===3||t===4,this.attributeName=r,this.attributeNamespace=i,this.mustUseProperty=n,this.propertyName=e,this.type=t,this.sanitizeURL=o,this.removeEmptyString=s}var ie={};"children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style".split(" ").forEach(function(e){ie[e]=new he(e,0,!1,e,null,!1,!1)});[["acceptCharset","accept-charset"],["className","class"],["htmlFor","for"],["httpEquiv","http-equiv"]].forEach(function(e){var t=e[0];ie[t]=new he(t,1,!1,e[1],null,!1,!1)});["contentEditable","draggable","spellCheck","value"].forEach(function(e){ie[e]=new he(e,2,!1,e.toLowerCase(),null,!1,!1)});["autoReverse","externalResourcesRequired","focusable","preserveAlpha"].forEach(function(e){ie[e]=new he(e,2,!1,e,null,!1,!1)});"allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope".split(" ").forEach(function(e){ie[e]=new he(e,3,!1,e.toLowerCase(),null,!1,!1)});["checked","multiple","muted","selected"].forEach(function(e){ie[e]=new he(e,3,!0,e,null,!1,!1)});["capture","download"].forEach(function(e){ie[e]=new he(e,4,!1,e,null,!1,!1)});["cols","rows","size","span"].forEach(function(e){ie[e]=new he(e,6,!1,e,null,!1,!1)});["rowSpan","start"].forEach(function(e){ie[e]=new he(e,5,!1,e.toLowerCase(),null,!1,!1)});var qa=/[\-:]([a-z])/g;function Qa(e){return e[1].toUpperCase()}"accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height".split(" ").forEach(function(e){var t=e.replace(qa,Qa);ie[t]=new he(t,1,!1,e,null,!1,!1)});"xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type".split(" ").forEach(function(e){var t=e.replace(qa,Qa);ie[t]=new he(t,1,!1,e,"http://www.w3.org/1999/xlink",!1,!1)});["xml:base","xml:lang","xml:space"].forEach(function(e){var t=e.replace(qa,Qa);ie[t]=new he(t,1,!1,e,"http://www.w3.org/XML/1998/namespace",!1,!1)});["tabIndex","crossOrigin"].forEach(function(e){ie[e]=new he(e,1,!1,e.toLowerCase(),null,!1,!1)});ie.xlinkHref=new he("xlinkHref",1,!1,"xlink:href","http://www.w3.org/1999/xlink",!0,!1);["src","href","action","formAction"].forEach(function(e){ie[e]=new he(e,1,!1,e.toLowerCase(),null,!0,!0)});function Ya(e,t,n,r){var i=ie.hasOwnProperty(t)?ie[t]:null;(i!==null?i.type!==0:r||!(2<t.length)||t[0]!=="o"&&t[0]!=="O"||t[1]!=="n"&&t[1]!=="N")&&(Am(t,n,i,r)&&(n=null),r||i===null?Nm(t)&&(n===null?e.removeAttribute(t):e.setAttribute(t,""+n)):i.mustUseProperty?e[i.propertyName]=n===null?i.type===3?!1:"":n:(t=i.attributeName,r=i.attributeNamespace,n===null?e.removeAttribute(t):(i=i.type,n=i===3||i===4&&n===!0?"":""+n,r?e.setAttributeNS(r,t,n):e.setAttribute(t,n))))}var mt=Pm.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,di=Symbol.for("react.element"),En=Symbol.for("react.portal"),In=Symbol.for("react.fragment"),Ja=Symbol.for("react.strict_mode"),bs=Symbol.for("react.profiler"),kd=Symbol.for("react.provider"),Cd=Symbol.for("react.context"),Xa=Symbol.for("react.forward_ref"),$s=Symbol.for("react.suspense"),Bs=Symbol.for("react.suspense_list"),Za=Symbol.for("react.memo"),wt=Symbol.for("react.lazy"),Td=Symbol.for("react.offscreen"),fu=Symbol.iterator;function ir(e){return e===null||typeof e!="object"?null:(e=fu&&e[fu]||e["@@iterator"],typeof e=="function"?e:null)}var W=Object.assign,ns;function fr(e){if(ns===void 0)try{throw Error()}catch(n){var t=n.stack.trim().match(/\n( *(at )?)/);ns=t&&t[1]||""}return`
`+ns+e}var rs=!1;function is(e,t){if(!e||rs)return"";rs=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{if(t)if(t=function(){throw Error()},Object.defineProperty(t.prototype,"props",{set:function(){throw Error()}}),typeof Reflect=="object"&&Reflect.construct){try{Reflect.construct(t,[])}catch(u){var r=u}Reflect.construct(e,[],t)}else{try{t.call()}catch(u){r=u}e.call(t.prototype)}else{try{throw Error()}catch(u){r=u}e()}}catch(u){if(u&&r&&typeof u.stack=="string"){for(var i=u.stack.split(`
`),o=r.stack.split(`
`),s=i.length-1,a=o.length-1;1<=s&&0<=a&&i[s]!==o[a];)a--;for(;1<=s&&0<=a;s--,a--)if(i[s]!==o[a]){if(s!==1||a!==1)do if(s--,a--,0>a||i[s]!==o[a]){var l=`
`+i[s].replace(" at new "," at ");return e.displayName&&l.includes("<anonymous>")&&(l=l.replace("<anonymous>",e.displayName)),l}while(1<=s&&0<=a);break}}}finally{rs=!1,Error.prepareStackTrace=n}return(e=e?e.displayName||e.name:"")?fr(e):""}function xm(e){switch(e.tag){case 5:return fr(e.type);case 16:return fr("Lazy");case 13:return fr("Suspense");case 19:return fr("SuspenseList");case 0:case 2:case 15:return e=is(e.type,!1),e;case 11:return e=is(e.type.render,!1),e;case 1:return e=is(e.type,!0),e;default:return""}}function Vs(e){if(e==null)return null;if(typeof e=="function")return e.displayName||e.name||null;if(typeof e=="string")return e;switch(e){case In:return"Fragment";case En:return"Portal";case bs:return"Profiler";case Ja:return"StrictMode";case $s:return"Suspense";case Bs:return"SuspenseList"}if(typeof e=="object")switch(e.$$typeof){case Cd:return(e.displayName||"Context")+".Consumer";case kd:return(e._context.displayName||"Context")+".Provider";case Xa:var t=e.render;return e=e.displayName,e||(e=t.displayName||t.name||"",e=e!==""?"ForwardRef("+e+")":"ForwardRef"),e;case Za:return t=e.displayName||null,t!==null?t:Vs(e.type)||"Memo";case wt:t=e._payload,e=e._init;try{return Vs(e(t))}catch{}}return null}function Lm(e){var t=e.type;switch(e.tag){case 24:return"Cache";case 9:return(t.displayName||"Context")+".Consumer";case 10:return(t._context.displayName||"Context")+".Provider";case 18:return"DehydratedFragment";case 11:return e=t.render,e=e.displayName||e.name||"",t.displayName||(e!==""?"ForwardRef("+e+")":"ForwardRef");case 7:return"Fragment";case 5:return t;case 4:return"Portal";case 3:return"Root";case 6:return"Text";case 16:return Vs(t);case 8:return t===Ja?"StrictMode":"Mode";case 22:return"Offscreen";case 12:return"Profiler";case 21:return"Scope";case 13:return"Suspense";case 19:return"SuspenseList";case 25:return"TracingMarker";case 1:case 0:case 17:case 2:case 14:case 15:if(typeof t=="function")return t.displayName||t.name||null;if(typeof t=="string")return t}return null}function $t(e){switch(typeof e){case"boolean":case"number":case"string":case"undefined":return e;case"object":return e;default:return""}}function Pd(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()==="input"&&(t==="checkbox"||t==="radio")}function Dm(e){var t=Pd(e)?"checked":"value",n=Object.getOwnPropertyDescriptor(e.constructor.prototype,t),r=""+e[t];if(!e.hasOwnProperty(t)&&typeof n<"u"&&typeof n.get=="function"&&typeof n.set=="function"){var i=n.get,o=n.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(s){r=""+s,o.call(this,s)}}),Object.defineProperty(e,t,{enumerable:n.enumerable}),{getValue:function(){return r},setValue:function(s){r=""+s},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function fi(e){e._valueTracker||(e._valueTracker=Dm(e))}function Rd(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r="";return e&&(r=Pd(e)?e.checked?"true":"false":e.value),e=r,e!==n?(t.setValue(e),!0):!1}function Hi(e){if(e=e||(typeof document<"u"?document:void 0),typeof e>"u")return null;try{return e.activeElement||e.body}catch{return e.body}}function Hs(e,t){var n=t.checked;return W({},t,{defaultChecked:void 0,defaultValue:void 0,value:void 0,checked:n??e._wrapperState.initialChecked})}function hu(e,t){var n=t.defaultValue==null?"":t.defaultValue,r=t.checked!=null?t.checked:t.defaultChecked;n=$t(t.value!=null?t.value:n),e._wrapperState={initialChecked:r,initialValue:n,controlled:t.type==="checkbox"||t.type==="radio"?t.checked!=null:t.value!=null}}function Nd(e,t){t=t.checked,t!=null&&Ya(e,"checked",t,!1)}function Ws(e,t){Nd(e,t);var n=$t(t.value),r=t.type;if(n!=null)r==="number"?(n===0&&e.value===""||e.value!=n)&&(e.value=""+n):e.value!==""+n&&(e.value=""+n);else if(r==="submit"||r==="reset"){e.removeAttribute("value");return}t.hasOwnProperty("value")?Ks(e,t.type,n):t.hasOwnProperty("defaultValue")&&Ks(e,t.type,$t(t.defaultValue)),t.checked==null&&t.defaultChecked!=null&&(e.defaultChecked=!!t.defaultChecked)}function pu(e,t,n){if(t.hasOwnProperty("value")||t.hasOwnProperty("defaultValue")){var r=t.type;if(!(r!=="submit"&&r!=="reset"||t.value!==void 0&&t.value!==null))return;t=""+e._wrapperState.initialValue,n||t===e.value||(e.value=t),e.defaultValue=t}n=e.name,n!==""&&(e.name=""),e.defaultChecked=!!e._wrapperState.initialChecked,n!==""&&(e.name=n)}function Ks(e,t,n){(t!=="number"||Hi(e.ownerDocument)!==e)&&(n==null?e.defaultValue=""+e._wrapperState.initialValue:e.defaultValue!==""+n&&(e.defaultValue=""+n))}var hr=Array.isArray;function Mn(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t["$"+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty("$"+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=""+$t(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function Gs(e,t){if(t.dangerouslySetInnerHTML!=null)throw Error(_(91));return W({},t,{value:void 0,defaultValue:void 0,children:""+e._wrapperState.initialValue})}function mu(e,t){var n=t.value;if(n==null){if(n=t.children,t=t.defaultValue,n!=null){if(t!=null)throw Error(_(92));if(hr(n)){if(1<n.length)throw Error(_(93));n=n[0]}t=n}t==null&&(t=""),n=t}e._wrapperState={initialValue:$t(n)}}function Od(e,t){var n=$t(t.value),r=$t(t.defaultValue);n!=null&&(n=""+n,n!==e.value&&(e.value=n),t.defaultValue==null&&e.defaultValue!==n&&(e.defaultValue=n)),r!=null&&(e.defaultValue=""+r)}function gu(e){var t=e.textContent;t===e._wrapperState.initialValue&&t!==""&&t!==null&&(e.value=t)}function Ad(e){switch(e){case"svg":return"http://www.w3.org/2000/svg";case"math":return"http://www.w3.org/1998/Math/MathML";default:return"http://www.w3.org/1999/xhtml"}}function qs(e,t){return e==null||e==="http://www.w3.org/1999/xhtml"?Ad(t):e==="http://www.w3.org/2000/svg"&&t==="foreignObject"?"http://www.w3.org/1999/xhtml":e}var hi,xd=function(e){return typeof MSApp<"u"&&MSApp.execUnsafeLocalFunction?function(t,n,r,i){MSApp.execUnsafeLocalFunction(function(){return e(t,n,r,i)})}:e}(function(e,t){if(e.namespaceURI!=="http://www.w3.org/2000/svg"||"innerHTML"in e)e.innerHTML=t;else{for(hi=hi||document.createElement("div"),hi.innerHTML="<svg>"+t.valueOf().toString()+"</svg>",t=hi.firstChild;e.firstChild;)e.removeChild(e.firstChild);for(;t.firstChild;)e.appendChild(t.firstChild)}});function Nr(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var vr={animationIterationCount:!0,aspectRatio:!0,borderImageOutset:!0,borderImageSlice:!0,borderImageWidth:!0,boxFlex:!0,boxFlexGroup:!0,boxOrdinalGroup:!0,columnCount:!0,columns:!0,flex:!0,flexGrow:!0,flexPositive:!0,flexShrink:!0,flexNegative:!0,flexOrder:!0,gridArea:!0,gridRow:!0,gridRowEnd:!0,gridRowSpan:!0,gridRowStart:!0,gridColumn:!0,gridColumnEnd:!0,gridColumnSpan:!0,gridColumnStart:!0,fontWeight:!0,lineClamp:!0,lineHeight:!0,opacity:!0,order:!0,orphans:!0,tabSize:!0,widows:!0,zIndex:!0,zoom:!0,fillOpacity:!0,floodOpacity:!0,stopOpacity:!0,strokeDasharray:!0,strokeDashoffset:!0,strokeMiterlimit:!0,strokeOpacity:!0,strokeWidth:!0},Mm=["Webkit","ms","Moz","O"];Object.keys(vr).forEach(function(e){Mm.forEach(function(t){t=t+e.charAt(0).toUpperCase()+e.substring(1),vr[t]=vr[e]})});function Ld(e,t,n){return t==null||typeof t=="boolean"||t===""?"":n||typeof t!="number"||t===0||vr.hasOwnProperty(e)&&vr[e]?(""+t).trim():t+"px"}function Dd(e,t){e=e.style;for(var n in t)if(t.hasOwnProperty(n)){var r=n.indexOf("--")===0,i=Ld(n,t[n],r);n==="float"&&(n="cssFloat"),r?e.setProperty(n,i):e[n]=i}}var Um=W({menuitem:!0},{area:!0,base:!0,br:!0,col:!0,embed:!0,hr:!0,img:!0,input:!0,keygen:!0,link:!0,meta:!0,param:!0,source:!0,track:!0,wbr:!0});function Qs(e,t){if(t){if(Um[e]&&(t.children!=null||t.dangerouslySetInnerHTML!=null))throw Error(_(137,e));if(t.dangerouslySetInnerHTML!=null){if(t.children!=null)throw Error(_(60));if(typeof t.dangerouslySetInnerHTML!="object"||!("__html"in t.dangerouslySetInnerHTML))throw Error(_(61))}if(t.style!=null&&typeof t.style!="object")throw Error(_(62))}}function Ys(e,t){if(e.indexOf("-")===-1)return typeof t.is=="string";switch(e){case"annotation-xml":case"color-profile":case"font-face":case"font-face-src":case"font-face-uri":case"font-face-format":case"font-face-name":case"missing-glyph":return!1;default:return!0}}var Js=null;function el(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Xs=null,Un=null,jn=null;function vu(e){if(e=Zr(e)){if(typeof Xs!="function")throw Error(_(280));var t=e.stateNode;t&&(t=xo(t),Xs(e.stateNode,e.type,t))}}function Md(e){Un?jn?jn.push(e):jn=[e]:Un=e}function Ud(){if(Un){var e=Un,t=jn;if(jn=Un=null,vu(e),t)for(e=0;e<t.length;e++)vu(t[e])}}function jd(e,t){return e(t)}function Fd(){}var os=!1;function zd(e,t,n){if(os)return e(t,n);os=!0;try{return jd(e,t,n)}finally{os=!1,(Un!==null||jn!==null)&&(Fd(),Ud())}}function Or(e,t){var n=e.stateNode;if(n===null)return null;var r=xo(n);if(r===null)return null;n=r[t];e:switch(t){case"onClick":case"onClickCapture":case"onDoubleClick":case"onDoubleClickCapture":case"onMouseDown":case"onMouseDownCapture":case"onMouseMove":case"onMouseMoveCapture":case"onMouseUp":case"onMouseUpCapture":case"onMouseEnter":(r=!r.disabled)||(e=e.type,r=!(e==="button"||e==="input"||e==="select"||e==="textarea")),e=!r;break e;default:e=!1}if(e)return null;if(n&&typeof n!="function")throw Error(_(231,t,typeof n));return n}var Zs=!1;if(ut)try{var or={};Object.defineProperty(or,"passive",{get:function(){Zs=!0}}),window.addEventListener("test",or,or),window.removeEventListener("test",or,or)}catch{Zs=!1}function jm(e,t,n,r,i,o,s,a,l){var u=Array.prototype.slice.call(arguments,3);try{t.apply(n,u)}catch(d){this.onError(d)}}var yr=!1,Wi=null,Ki=!1,ea=null,Fm={onError:function(e){yr=!0,Wi=e}};function zm(e,t,n,r,i,o,s,a,l){yr=!1,Wi=null,jm.apply(Fm,arguments)}function bm(e,t,n,r,i,o,s,a,l){if(zm.apply(this,arguments),yr){if(yr){var u=Wi;yr=!1,Wi=null}else throw Error(_(198));Ki||(Ki=!0,ea=u)}}function mn(e){var t=e,n=e;if(e.alternate)for(;t.return;)t=t.return;else{e=t;do t=e,t.flags&4098&&(n=t.return),e=t.return;while(e)}return t.tag===3?n:null}function bd(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function yu(e){if(mn(e)!==e)throw Error(_(188))}function $m(e){var t=e.alternate;if(!t){if(t=mn(e),t===null)throw Error(_(188));return t!==e?null:e}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var o=i.alternate;if(o===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===o.child){for(o=i.child;o;){if(o===n)return yu(i),e;if(o===r)return yu(i),t;o=o.sibling}throw Error(_(188))}if(n.return!==r.return)n=i,r=o;else{for(var s=!1,a=i.child;a;){if(a===n){s=!0,n=i,r=o;break}if(a===r){s=!0,r=i,n=o;break}a=a.sibling}if(!s){for(a=o.child;a;){if(a===n){s=!0,n=o,r=i;break}if(a===r){s=!0,r=o,n=i;break}a=a.sibling}if(!s)throw Error(_(189))}}if(n.alternate!==r)throw Error(_(190))}if(n.tag!==3)throw Error(_(188));return n.stateNode.current===n?e:t}function $d(e){return e=$m(e),e!==null?Bd(e):null}function Bd(e){if(e.tag===5||e.tag===6)return e;for(e=e.child;e!==null;){var t=Bd(e);if(t!==null)return t;e=e.sibling}return null}var Vd=ke.unstable_scheduleCallback,wu=ke.unstable_cancelCallback,Bm=ke.unstable_shouldYield,Vm=ke.unstable_requestPaint,q=ke.unstable_now,Hm=ke.unstable_getCurrentPriorityLevel,tl=ke.unstable_ImmediatePriority,Hd=ke.unstable_UserBlockingPriority,Gi=ke.unstable_NormalPriority,Wm=ke.unstable_LowPriority,Wd=ke.unstable_IdlePriority,Ro=null,We=null;function Km(e){if(We&&typeof We.onCommitFiberRoot=="function")try{We.onCommitFiberRoot(Ro,e,void 0,(e.current.flags&128)===128)}catch{}}var Fe=Math.clz32?Math.clz32:Qm,Gm=Math.log,qm=Math.LN2;function Qm(e){return e>>>=0,e===0?32:31-(Gm(e)/qm|0)|0}var pi=64,mi=4194304;function pr(e){switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return e&4194240;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return e&130023424;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 1073741824;default:return e}}function qi(e,t){var n=e.pendingLanes;if(n===0)return 0;var r=0,i=e.suspendedLanes,o=e.pingedLanes,s=n&268435455;if(s!==0){var a=s&~i;a!==0?r=pr(a):(o&=s,o!==0&&(r=pr(o)))}else s=n&~i,s!==0?r=pr(s):o!==0&&(r=pr(o));if(r===0)return 0;if(t!==0&&t!==r&&!(t&i)&&(i=r&-r,o=t&-t,i>=o||i===16&&(o&4194240)!==0))return t;if(r&4&&(r|=n&16),t=e.entangledLanes,t!==0)for(e=e.entanglements,t&=r;0<t;)n=31-Fe(t),i=1<<n,r|=e[n],t&=~i;return r}function Ym(e,t){switch(e){case 1:case 2:case 4:return t+250;case 8:case 16:case 32:case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:return-1;case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function Jm(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,o=e.pendingLanes;0<o;){var s=31-Fe(o),a=1<<s,l=i[s];l===-1?(!(a&n)||a&r)&&(i[s]=Ym(a,t)):l<=t&&(e.expiredLanes|=a),o&=~a}}function ta(e){return e=e.pendingLanes&-1073741825,e!==0?e:e&1073741824?1073741824:0}function Kd(){var e=pi;return pi<<=1,!(pi&4194240)&&(pi=64),e}function ss(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function Jr(e,t,n){e.pendingLanes|=t,t!==536870912&&(e.suspendedLanes=0,e.pingedLanes=0),e=e.eventTimes,t=31-Fe(t),e[t]=n}function Xm(e,t){var n=e.pendingLanes&~t;e.pendingLanes=t,e.suspendedLanes=0,e.pingedLanes=0,e.expiredLanes&=t,e.mutableReadLanes&=t,e.entangledLanes&=t,t=e.entanglements;var r=e.eventTimes;for(e=e.expirationTimes;0<n;){var i=31-Fe(n),o=1<<i;t[i]=0,r[i]=-1,e[i]=-1,n&=~o}}function nl(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Fe(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}var U=0;function Gd(e){return e&=-e,1<e?4<e?e&268435455?16:536870912:4:1}var qd,rl,Qd,Yd,Jd,na=!1,gi=[],Ot=null,At=null,xt=null,Ar=new Map,xr=new Map,St=[],Zm="mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit".split(" ");function _u(e,t){switch(e){case"focusin":case"focusout":Ot=null;break;case"dragenter":case"dragleave":At=null;break;case"mouseover":case"mouseout":xt=null;break;case"pointerover":case"pointerout":Ar.delete(t.pointerId);break;case"gotpointercapture":case"lostpointercapture":xr.delete(t.pointerId)}}function sr(e,t,n,r,i,o){return e===null||e.nativeEvent!==o?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:o,targetContainers:[i]},t!==null&&(t=Zr(t),t!==null&&rl(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function eg(e,t,n,r,i){switch(t){case"focusin":return Ot=sr(Ot,e,t,n,r,i),!0;case"dragenter":return At=sr(At,e,t,n,r,i),!0;case"mouseover":return xt=sr(xt,e,t,n,r,i),!0;case"pointerover":var o=i.pointerId;return Ar.set(o,sr(Ar.get(o)||null,e,t,n,r,i)),!0;case"gotpointercapture":return o=i.pointerId,xr.set(o,sr(xr.get(o)||null,e,t,n,r,i)),!0}return!1}function Xd(e){var t=Jt(e.target);if(t!==null){var n=mn(t);if(n!==null){if(t=n.tag,t===13){if(t=bd(n),t!==null){e.blockedOn=t,Jd(e.priority,function(){Qd(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Oi(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=ra(e.domEventName,e.eventSystemFlags,t[0],e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);Js=r,n.target.dispatchEvent(r),Js=null}else return t=Zr(n),t!==null&&rl(t),e.blockedOn=n,!1;t.shift()}return!0}function Su(e,t,n){Oi(e)&&n.delete(t)}function tg(){na=!1,Ot!==null&&Oi(Ot)&&(Ot=null),At!==null&&Oi(At)&&(At=null),xt!==null&&Oi(xt)&&(xt=null),Ar.forEach(Su),xr.forEach(Su)}function ar(e,t){e.blockedOn===t&&(e.blockedOn=null,na||(na=!0,ke.unstable_scheduleCallback(ke.unstable_NormalPriority,tg)))}function Lr(e){function t(i){return ar(i,e)}if(0<gi.length){ar(gi[0],e);for(var n=1;n<gi.length;n++){var r=gi[n];r.blockedOn===e&&(r.blockedOn=null)}}for(Ot!==null&&ar(Ot,e),At!==null&&ar(At,e),xt!==null&&ar(xt,e),Ar.forEach(t),xr.forEach(t),n=0;n<St.length;n++)r=St[n],r.blockedOn===e&&(r.blockedOn=null);for(;0<St.length&&(n=St[0],n.blockedOn===null);)Xd(n),n.blockedOn===null&&St.shift()}var Fn=mt.ReactCurrentBatchConfig,Qi=!0;function ng(e,t,n,r){var i=U,o=Fn.transition;Fn.transition=null;try{U=1,il(e,t,n,r)}finally{U=i,Fn.transition=o}}function rg(e,t,n,r){var i=U,o=Fn.transition;Fn.transition=null;try{U=4,il(e,t,n,r)}finally{U=i,Fn.transition=o}}function il(e,t,n,r){if(Qi){var i=ra(e,t,n,r);if(i===null)gs(e,t,r,Yi,n),_u(e,r);else if(eg(i,e,t,n,r))r.stopPropagation();else if(_u(e,r),t&4&&-1<Zm.indexOf(e)){for(;i!==null;){var o=Zr(i);if(o!==null&&qd(o),o=ra(e,t,n,r),o===null&&gs(e,t,r,Yi,n),o===i)break;i=o}i!==null&&r.stopPropagation()}else gs(e,t,r,null,n)}}var Yi=null;function ra(e,t,n,r){if(Yi=null,e=el(r),e=Jt(e),e!==null)if(t=mn(e),t===null)e=null;else if(n=t.tag,n===13){if(e=bd(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null);return Yi=e,null}function Zd(e){switch(e){case"cancel":case"click":case"close":case"contextmenu":case"copy":case"cut":case"auxclick":case"dblclick":case"dragend":case"dragstart":case"drop":case"focusin":case"focusout":case"input":case"invalid":case"keydown":case"keypress":case"keyup":case"mousedown":case"mouseup":case"paste":case"pause":case"play":case"pointercancel":case"pointerdown":case"pointerup":case"ratechange":case"reset":case"resize":case"seeked":case"submit":case"touchcancel":case"touchend":case"touchstart":case"volumechange":case"change":case"selectionchange":case"textInput":case"compositionstart":case"compositionend":case"compositionupdate":case"beforeblur":case"afterblur":case"beforeinput":case"blur":case"fullscreenchange":case"focus":case"hashchange":case"popstate":case"select":case"selectstart":return 1;case"drag":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"mousemove":case"mouseout":case"mouseover":case"pointermove":case"pointerout":case"pointerover":case"scroll":case"toggle":case"touchmove":case"wheel":case"mouseenter":case"mouseleave":case"pointerenter":case"pointerleave":return 4;case"message":switch(Hm()){case tl:return 1;case Hd:return 4;case Gi:case Wm:return 16;case Wd:return 536870912;default:return 16}default:return 16}}var Pt=null,ol=null,Ai=null;function ef(){if(Ai)return Ai;var e,t=ol,n=t.length,r,i="value"in Pt?Pt.value:Pt.textContent,o=i.length;for(e=0;e<n&&t[e]===i[e];e++);var s=n-e;for(r=1;r<=s&&t[n-r]===i[o-r];r++);return Ai=i.slice(e,1<r?1-r:void 0)}function xi(e){var t=e.keyCode;return"charCode"in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function vi(){return!0}function Eu(){return!1}function Te(e){function t(n,r,i,o,s){this._reactName=n,this._targetInst=i,this.type=r,this.nativeEvent=o,this.target=s,this.currentTarget=null;for(var a in e)e.hasOwnProperty(a)&&(n=e[a],this[a]=n?n(o):o[a]);return this.isDefaultPrevented=(o.defaultPrevented!=null?o.defaultPrevented:o.returnValue===!1)?vi:Eu,this.isPropagationStopped=Eu,this}return W(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var n=this.nativeEvent;n&&(n.preventDefault?n.preventDefault():typeof n.returnValue!="unknown"&&(n.returnValue=!1),this.isDefaultPrevented=vi)},stopPropagation:function(){var n=this.nativeEvent;n&&(n.stopPropagation?n.stopPropagation():typeof n.cancelBubble!="unknown"&&(n.cancelBubble=!0),this.isPropagationStopped=vi)},persist:function(){},isPersistent:vi}),t}var Xn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},sl=Te(Xn),Xr=W({},Xn,{view:0,detail:0}),ig=Te(Xr),as,ls,lr,No=W({},Xr,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:al,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return"movementX"in e?e.movementX:(e!==lr&&(lr&&e.type==="mousemove"?(as=e.screenX-lr.screenX,ls=e.screenY-lr.screenY):ls=as=0,lr=e),as)},movementY:function(e){return"movementY"in e?e.movementY:ls}}),Iu=Te(No),og=W({},No,{dataTransfer:0}),sg=Te(og),ag=W({},Xr,{relatedTarget:0}),us=Te(ag),lg=W({},Xn,{animationName:0,elapsedTime:0,pseudoElement:0}),ug=Te(lg),cg=W({},Xn,{clipboardData:function(e){return"clipboardData"in e?e.clipboardData:window.clipboardData}}),dg=Te(cg),fg=W({},Xn,{data:0}),ku=Te(fg),hg={Esc:"Escape",Spacebar:" ",Left:"ArrowLeft",Up:"ArrowUp",Right:"ArrowRight",Down:"ArrowDown",Del:"Delete",Win:"OS",Menu:"ContextMenu",Apps:"ContextMenu",Scroll:"ScrollLock",MozPrintableKey:"Unidentified"},pg={8:"Backspace",9:"Tab",12:"Clear",13:"Enter",16:"Shift",17:"Control",18:"Alt",19:"Pause",20:"CapsLock",27:"Escape",32:" ",33:"PageUp",34:"PageDown",35:"End",36:"Home",37:"ArrowLeft",38:"ArrowUp",39:"ArrowRight",40:"ArrowDown",45:"Insert",46:"Delete",112:"F1",113:"F2",114:"F3",115:"F4",116:"F5",117:"F6",118:"F7",119:"F8",120:"F9",121:"F10",122:"F11",123:"F12",144:"NumLock",145:"ScrollLock",224:"Meta"},mg={Alt:"altKey",Control:"ctrlKey",Meta:"metaKey",Shift:"shiftKey"};function gg(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=mg[e])?!!t[e]:!1}function al(){return gg}var vg=W({},Xr,{key:function(e){if(e.key){var t=hg[e.key]||e.key;if(t!=="Unidentified")return t}return e.type==="keypress"?(e=xi(e),e===13?"Enter":String.fromCharCode(e)):e.type==="keydown"||e.type==="keyup"?pg[e.keyCode]||"Unidentified":""},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:al,charCode:function(e){return e.type==="keypress"?xi(e):0},keyCode:function(e){return e.type==="keydown"||e.type==="keyup"?e.keyCode:0},which:function(e){return e.type==="keypress"?xi(e):e.type==="keydown"||e.type==="keyup"?e.keyCode:0}}),yg=Te(vg),wg=W({},No,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0}),Cu=Te(wg),_g=W({},Xr,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:al}),Sg=Te(_g),Eg=W({},Xn,{propertyName:0,elapsedTime:0,pseudoElement:0}),Ig=Te(Eg),kg=W({},No,{deltaX:function(e){return"deltaX"in e?e.deltaX:"wheelDeltaX"in e?-e.wheelDeltaX:0},deltaY:function(e){return"deltaY"in e?e.deltaY:"wheelDeltaY"in e?-e.wheelDeltaY:"wheelDelta"in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0}),Cg=Te(kg),Tg=[9,13,27,32],ll=ut&&"CompositionEvent"in window,wr=null;ut&&"documentMode"in document&&(wr=document.documentMode);var Pg=ut&&"TextEvent"in window&&!wr,tf=ut&&(!ll||wr&&8<wr&&11>=wr),Tu=" ",Pu=!1;function nf(e,t){switch(e){case"keyup":return Tg.indexOf(t.keyCode)!==-1;case"keydown":return t.keyCode!==229;case"keypress":case"mousedown":case"focusout":return!0;default:return!1}}function rf(e){return e=e.detail,typeof e=="object"&&"data"in e?e.data:null}var kn=!1;function Rg(e,t){switch(e){case"compositionend":return rf(t);case"keypress":return t.which!==32?null:(Pu=!0,Tu);case"textInput":return e=t.data,e===Tu&&Pu?null:e;default:return null}}function Ng(e,t){if(kn)return e==="compositionend"||!ll&&nf(e,t)?(e=ef(),Ai=ol=Pt=null,kn=!1,e):null;switch(e){case"paste":return null;case"keypress":if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case"compositionend":return tf&&t.locale!=="ko"?null:t.data;default:return null}}var Og={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Ru(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t==="input"?!!Og[e.type]:t==="textarea"}function of(e,t,n,r){Md(r),t=Ji(t,"onChange"),0<t.length&&(n=new sl("onChange","change",null,n,r),e.push({event:n,listeners:t}))}var _r=null,Dr=null;function Ag(e){gf(e,0)}function Oo(e){var t=Pn(e);if(Rd(t))return e}function xg(e,t){if(e==="change")return t}var sf=!1;if(ut){var cs;if(ut){var ds="oninput"in document;if(!ds){var Nu=document.createElement("div");Nu.setAttribute("oninput","return;"),ds=typeof Nu.oninput=="function"}cs=ds}else cs=!1;sf=cs&&(!document.documentMode||9<document.documentMode)}function Ou(){_r&&(_r.detachEvent("onpropertychange",af),Dr=_r=null)}function af(e){if(e.propertyName==="value"&&Oo(Dr)){var t=[];of(t,Dr,e,el(e)),zd(Ag,t)}}function Lg(e,t,n){e==="focusin"?(Ou(),_r=t,Dr=n,_r.attachEvent("onpropertychange",af)):e==="focusout"&&Ou()}function Dg(e){if(e==="selectionchange"||e==="keyup"||e==="keydown")return Oo(Dr)}function Mg(e,t){if(e==="click")return Oo(t)}function Ug(e,t){if(e==="input"||e==="change")return Oo(t)}function jg(e,t){return e===t&&(e!==0||1/e===1/t)||e!==e&&t!==t}var be=typeof Object.is=="function"?Object.is:jg;function Mr(e,t){if(be(e,t))return!0;if(typeof e!="object"||e===null||typeof t!="object"||t===null)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!zs.call(t,i)||!be(e[i],t[i]))return!1}return!0}function Au(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function xu(e,t){var n=Au(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}e:{for(;n;){if(n.nextSibling){n=n.nextSibling;break e}n=n.parentNode}n=void 0}n=Au(n)}}function lf(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?lf(e,t.parentNode):"contains"in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function uf(){for(var e=window,t=Hi();t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href=="string"}catch{n=!1}if(n)e=t.contentWindow;else break;t=Hi(e.document)}return t}function ul(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t==="input"&&(e.type==="text"||e.type==="search"||e.type==="tel"||e.type==="url"||e.type==="password")||t==="textarea"||e.contentEditable==="true")}function Fg(e){var t=uf(),n=e.focusedElem,r=e.selectionRange;if(t!==n&&n&&n.ownerDocument&&lf(n.ownerDocument.documentElement,n)){if(r!==null&&ul(n)){if(t=r.start,e=r.end,e===void 0&&(e=t),"selectionStart"in n)n.selectionStart=t,n.selectionEnd=Math.min(e,n.value.length);else if(e=(t=n.ownerDocument||document)&&t.defaultView||window,e.getSelection){e=e.getSelection();var i=n.textContent.length,o=Math.min(r.start,i);r=r.end===void 0?o:Math.min(r.end,i),!e.extend&&o>r&&(i=r,r=o,o=i),i=xu(n,o);var s=xu(n,r);i&&s&&(e.rangeCount!==1||e.anchorNode!==i.node||e.anchorOffset!==i.offset||e.focusNode!==s.node||e.focusOffset!==s.offset)&&(t=t.createRange(),t.setStart(i.node,i.offset),e.removeAllRanges(),o>r?(e.addRange(t),e.extend(s.node,s.offset)):(t.setEnd(s.node,s.offset),e.addRange(t)))}}for(t=[],e=n;e=e.parentNode;)e.nodeType===1&&t.push({element:e,left:e.scrollLeft,top:e.scrollTop});for(typeof n.focus=="function"&&n.focus(),n=0;n<t.length;n++)e=t[n],e.element.scrollLeft=e.left,e.element.scrollTop=e.top}}var zg=ut&&"documentMode"in document&&11>=document.documentMode,Cn=null,ia=null,Sr=null,oa=!1;function Lu(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;oa||Cn==null||Cn!==Hi(r)||(r=Cn,"selectionStart"in r&&ul(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Sr&&Mr(Sr,r)||(Sr=r,r=Ji(ia,"onSelect"),0<r.length&&(t=new sl("onSelect","select",null,t,n),e.push({event:t,listeners:r}),t.target=Cn)))}function yi(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n["Webkit"+e]="webkit"+t,n["Moz"+e]="moz"+t,n}var Tn={animationend:yi("Animation","AnimationEnd"),animationiteration:yi("Animation","AnimationIteration"),animationstart:yi("Animation","AnimationStart"),transitionend:yi("Transition","TransitionEnd")},fs={},cf={};ut&&(cf=document.createElement("div").style,"AnimationEvent"in window||(delete Tn.animationend.animation,delete Tn.animationiteration.animation,delete Tn.animationstart.animation),"TransitionEvent"in window||delete Tn.transitionend.transition);function Ao(e){if(fs[e])return fs[e];if(!Tn[e])return e;var t=Tn[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in cf)return fs[e]=t[n];return e}var df=Ao("animationend"),ff=Ao("animationiteration"),hf=Ao("animationstart"),pf=Ao("transitionend"),mf=new Map,Du="abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel".split(" ");function Vt(e,t){mf.set(e,t),pn(t,[e])}for(var hs=0;hs<Du.length;hs++){var ps=Du[hs],bg=ps.toLowerCase(),$g=ps[0].toUpperCase()+ps.slice(1);Vt(bg,"on"+$g)}Vt(df,"onAnimationEnd");Vt(ff,"onAnimationIteration");Vt(hf,"onAnimationStart");Vt("dblclick","onDoubleClick");Vt("focusin","onFocus");Vt("focusout","onBlur");Vt(pf,"onTransitionEnd");Hn("onMouseEnter",["mouseout","mouseover"]);Hn("onMouseLeave",["mouseout","mouseover"]);Hn("onPointerEnter",["pointerout","pointerover"]);Hn("onPointerLeave",["pointerout","pointerover"]);pn("onChange","change click focusin focusout input keydown keyup selectionchange".split(" "));pn("onSelect","focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange".split(" "));pn("onBeforeInput",["compositionend","keypress","textInput","paste"]);pn("onCompositionEnd","compositionend focusout keydown keypress keyup mousedown".split(" "));pn("onCompositionStart","compositionstart focusout keydown keypress keyup mousedown".split(" "));pn("onCompositionUpdate","compositionupdate focusout keydown keypress keyup mousedown".split(" "));var mr="abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting".split(" "),Bg=new Set("cancel close invalid load scroll toggle".split(" ").concat(mr));function Mu(e,t,n){var r=e.type||"unknown-event";e.currentTarget=n,bm(r,t,void 0,e),e.currentTarget=null}function gf(e,t){t=(t&4)!==0;for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;e:{var o=void 0;if(t)for(var s=r.length-1;0<=s;s--){var a=r[s],l=a.instance,u=a.currentTarget;if(a=a.listener,l!==o&&i.isPropagationStopped())break e;Mu(i,a,u),o=l}else for(s=0;s<r.length;s++){if(a=r[s],l=a.instance,u=a.currentTarget,a=a.listener,l!==o&&i.isPropagationStopped())break e;Mu(i,a,u),o=l}}}if(Ki)throw e=ea,Ki=!1,ea=null,e}function b(e,t){var n=t[ca];n===void 0&&(n=t[ca]=new Set);var r=e+"__bubble";n.has(r)||(vf(t,e,2,!1),n.add(r))}function ms(e,t,n){var r=0;t&&(r|=4),vf(n,e,r,t)}var wi="_reactListening"+Math.random().toString(36).slice(2);function Ur(e){if(!e[wi]){e[wi]=!0,Id.forEach(function(n){n!=="selectionchange"&&(Bg.has(n)||ms(n,!1,e),ms(n,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[wi]||(t[wi]=!0,ms("selectionchange",!1,t))}}function vf(e,t,n,r){switch(Zd(t)){case 1:var i=ng;break;case 4:i=rg;break;default:i=il}n=i.bind(null,t,n,e),i=void 0,!Zs||t!=="touchstart"&&t!=="touchmove"&&t!=="wheel"||(i=!0),r?i!==void 0?e.addEventListener(t,n,{capture:!0,passive:i}):e.addEventListener(t,n,!0):i!==void 0?e.addEventListener(t,n,{passive:i}):e.addEventListener(t,n,!1)}function gs(e,t,n,r,i){var o=r;if(!(t&1)&&!(t&2)&&r!==null)e:for(;;){if(r===null)return;var s=r.tag;if(s===3||s===4){var a=r.stateNode.containerInfo;if(a===i||a.nodeType===8&&a.parentNode===i)break;if(s===4)for(s=r.return;s!==null;){var l=s.tag;if((l===3||l===4)&&(l=s.stateNode.containerInfo,l===i||l.nodeType===8&&l.parentNode===i))return;s=s.return}for(;a!==null;){if(s=Jt(a),s===null)return;if(l=s.tag,l===5||l===6){r=o=s;continue e}a=a.parentNode}}r=r.return}zd(function(){var u=o,d=el(n),f=[];e:{var m=mf.get(e);if(m!==void 0){var g=sl,y=e;switch(e){case"keypress":if(xi(n)===0)break e;case"keydown":case"keyup":g=yg;break;case"focusin":y="focus",g=us;break;case"focusout":y="blur",g=us;break;case"beforeblur":case"afterblur":g=us;break;case"click":if(n.button===2)break e;case"auxclick":case"dblclick":case"mousedown":case"mousemove":case"mouseup":case"mouseout":case"mouseover":case"contextmenu":g=Iu;break;case"drag":case"dragend":case"dragenter":case"dragexit":case"dragleave":case"dragover":case"dragstart":case"drop":g=sg;break;case"touchcancel":case"touchend":case"touchmove":case"touchstart":g=Sg;break;case df:case ff:case hf:g=ug;break;case pf:g=Ig;break;case"scroll":g=ig;break;case"wheel":g=Cg;break;case"copy":case"cut":case"paste":g=dg;break;case"gotpointercapture":case"lostpointercapture":case"pointercancel":case"pointerdown":case"pointermove":case"pointerout":case"pointerover":case"pointerup":g=Cu}var w=(t&4)!==0,E=!w&&e==="scroll",h=w?m!==null?m+"Capture":null:m;w=[];for(var c=u,p;c!==null;){p=c;var v=p.stateNode;if(p.tag===5&&v!==null&&(p=v,h!==null&&(v=Or(c,h),v!=null&&w.push(jr(c,v,p)))),E)break;c=c.return}0<w.length&&(m=new g(m,y,null,n,d),f.push({event:m,listeners:w}))}}if(!(t&7)){e:{if(m=e==="mouseover"||e==="pointerover",g=e==="mouseout"||e==="pointerout",m&&n!==Js&&(y=n.relatedTarget||n.fromElement)&&(Jt(y)||y[ct]))break e;if((g||m)&&(m=d.window===d?d:(m=d.ownerDocument)?m.defaultView||m.parentWindow:window,g?(y=n.relatedTarget||n.toElement,g=u,y=y?Jt(y):null,y!==null&&(E=mn(y),y!==E||y.tag!==5&&y.tag!==6)&&(y=null)):(g=null,y=u),g!==y)){if(w=Iu,v="onMouseLeave",h="onMouseEnter",c="mouse",(e==="pointerout"||e==="pointerover")&&(w=Cu,v="onPointerLeave",h="onPointerEnter",c="pointer"),E=g==null?m:Pn(g),p=y==null?m:Pn(y),m=new w(v,c+"leave",g,n,d),m.target=E,m.relatedTarget=p,v=null,Jt(d)===u&&(w=new w(h,c+"enter",y,n,d),w.target=p,w.relatedTarget=E,v=w),E=v,g&&y)t:{for(w=g,h=y,c=0,p=w;p;p=Sn(p))c++;for(p=0,v=h;v;v=Sn(v))p++;for(;0<c-p;)w=Sn(w),c--;for(;0<p-c;)h=Sn(h),p--;for(;c--;){if(w===h||h!==null&&w===h.alternate)break t;w=Sn(w),h=Sn(h)}w=null}else w=null;g!==null&&Uu(f,m,g,w,!1),y!==null&&E!==null&&Uu(f,E,y,w,!0)}}e:{if(m=u?Pn(u):window,g=m.nodeName&&m.nodeName.toLowerCase(),g==="select"||g==="input"&&m.type==="file")var S=xg;else if(Ru(m))if(sf)S=Ug;else{S=Dg;var P=Lg}else(g=m.nodeName)&&g.toLowerCase()==="input"&&(m.type==="checkbox"||m.type==="radio")&&(S=Mg);if(S&&(S=S(e,u))){of(f,S,n,d);break e}P&&P(e,m,u),e==="focusout"&&(P=m._wrapperState)&&P.controlled&&m.type==="number"&&Ks(m,"number",m.value)}switch(P=u?Pn(u):window,e){case"focusin":(Ru(P)||P.contentEditable==="true")&&(Cn=P,ia=u,Sr=null);break;case"focusout":Sr=ia=Cn=null;break;case"mousedown":oa=!0;break;case"contextmenu":case"mouseup":case"dragend":oa=!1,Lu(f,n,d);break;case"selectionchange":if(zg)break;case"keydown":case"keyup":Lu(f,n,d)}var R;if(ll)e:{switch(e){case"compositionstart":var N="onCompositionStart";break e;case"compositionend":N="onCompositionEnd";break e;case"compositionupdate":N="onCompositionUpdate";break e}N=void 0}else kn?nf(e,n)&&(N="onCompositionEnd"):e==="keydown"&&n.keyCode===229&&(N="onCompositionStart");N&&(tf&&n.locale!=="ko"&&(kn||N!=="onCompositionStart"?N==="onCompositionEnd"&&kn&&(R=ef()):(Pt=d,ol="value"in Pt?Pt.value:Pt.textContent,kn=!0)),P=Ji(u,N),0<P.length&&(N=new ku(N,e,null,n,d),f.push({event:N,listeners:P}),R?N.data=R:(R=rf(n),R!==null&&(N.data=R)))),(R=Pg?Rg(e,n):Ng(e,n))&&(u=Ji(u,"onBeforeInput"),0<u.length&&(d=new ku("onBeforeInput","beforeinput",null,n,d),f.push({event:d,listeners:u}),d.data=R))}gf(f,t)})}function jr(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Ji(e,t){for(var n=t+"Capture",r=[];e!==null;){var i=e,o=i.stateNode;i.tag===5&&o!==null&&(i=o,o=Or(e,n),o!=null&&r.unshift(jr(e,o,i)),o=Or(e,t),o!=null&&r.push(jr(e,o,i))),e=e.return}return r}function Sn(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5);return e||null}function Uu(e,t,n,r,i){for(var o=t._reactName,s=[];n!==null&&n!==r;){var a=n,l=a.alternate,u=a.stateNode;if(l!==null&&l===r)break;a.tag===5&&u!==null&&(a=u,i?(l=Or(n,o),l!=null&&s.unshift(jr(n,l,a))):i||(l=Or(n,o),l!=null&&s.push(jr(n,l,a)))),n=n.return}s.length!==0&&e.push({event:t,listeners:s})}var Vg=/\r\n?/g,Hg=/\u0000|\uFFFD/g;function ju(e){return(typeof e=="string"?e:""+e).replace(Vg,`
`).replace(Hg,"")}function _i(e,t,n){if(t=ju(t),ju(e)!==t&&n)throw Error(_(425))}function Xi(){}var sa=null,aa=null;function la(e,t){return e==="textarea"||e==="noscript"||typeof t.children=="string"||typeof t.children=="number"||typeof t.dangerouslySetInnerHTML=="object"&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var ua=typeof setTimeout=="function"?setTimeout:void 0,Wg=typeof clearTimeout=="function"?clearTimeout:void 0,Fu=typeof Promise=="function"?Promise:void 0,Kg=typeof queueMicrotask=="function"?queueMicrotask:typeof Fu<"u"?function(e){return Fu.resolve(null).then(e).catch(Gg)}:ua;function Gg(e){setTimeout(function(){throw e})}function vs(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8)if(n=i.data,n==="/$"){if(r===0){e.removeChild(i),Lr(t);return}r--}else n!=="$"&&n!=="$?"&&n!=="$!"||r++;n=i}while(n);Lr(t)}function Lt(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t==="$"||t==="$!"||t==="$?")break;if(t==="/$")return null}}return e}function zu(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="$"||n==="$!"||n==="$?"){if(t===0)return e;t--}else n==="/$"&&t++}e=e.previousSibling}return null}var Zn=Math.random().toString(36).slice(2),Ve="__reactFiber$"+Zn,Fr="__reactProps$"+Zn,ct="__reactContainer$"+Zn,ca="__reactEvents$"+Zn,qg="__reactListeners$"+Zn,Qg="__reactHandles$"+Zn;function Jt(e){var t=e[Ve];if(t)return t;for(var n=e.parentNode;n;){if(t=n[ct]||n[Ve]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=zu(e);e!==null;){if(n=e[Ve])return n;e=zu(e)}return t}e=n,n=e.parentNode}return null}function Zr(e){return e=e[Ve]||e[ct],!e||e.tag!==5&&e.tag!==6&&e.tag!==13&&e.tag!==3?null:e}function Pn(e){if(e.tag===5||e.tag===6)return e.stateNode;throw Error(_(33))}function xo(e){return e[Fr]||null}var da=[],Rn=-1;function Ht(e){return{current:e}}function $(e){0>Rn||(e.current=da[Rn],da[Rn]=null,Rn--)}function F(e,t){Rn++,da[Rn]=e.current,e.current=t}var Bt={},le=Ht(Bt),ge=Ht(!1),rn=Bt;function Wn(e,t){var n=e.type.contextTypes;if(!n)return Bt;var r=e.stateNode;if(r&&r.__reactInternalMemoizedUnmaskedChildContext===t)return r.__reactInternalMemoizedMaskedChildContext;var i={},o;for(o in n)i[o]=t[o];return r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=t,e.__reactInternalMemoizedMaskedChildContext=i),i}function ve(e){return e=e.childContextTypes,e!=null}function Zi(){$(ge),$(le)}function bu(e,t,n){if(le.current!==Bt)throw Error(_(168));F(le,t),F(ge,n)}function yf(e,t,n){var r=e.stateNode;if(t=t.childContextTypes,typeof r.getChildContext!="function")return n;r=r.getChildContext();for(var i in r)if(!(i in t))throw Error(_(108,Lm(e)||"Unknown",i));return W({},n,r)}function eo(e){return e=(e=e.stateNode)&&e.__reactInternalMemoizedMergedChildContext||Bt,rn=le.current,F(le,e),F(ge,ge.current),!0}function $u(e,t,n){var r=e.stateNode;if(!r)throw Error(_(169));n?(e=yf(e,t,rn),r.__reactInternalMemoizedMergedChildContext=e,$(ge),$(le),F(le,e)):$(ge),F(ge,n)}var et=null,Lo=!1,ys=!1;function wf(e){et===null?et=[e]:et.push(e)}function Yg(e){Lo=!0,wf(e)}function Wt(){if(!ys&&et!==null){ys=!0;var e=0,t=U;try{var n=et;for(U=1;e<n.length;e++){var r=n[e];do r=r(!0);while(r!==null)}et=null,Lo=!1}catch(i){throw et!==null&&(et=et.slice(e+1)),Vd(tl,Wt),i}finally{U=t,ys=!1}}return null}var Nn=[],On=0,to=null,no=0,Pe=[],Re=0,on=null,tt=1,nt="";function qt(e,t){Nn[On++]=no,Nn[On++]=to,to=e,no=t}function _f(e,t,n){Pe[Re++]=tt,Pe[Re++]=nt,Pe[Re++]=on,on=e;var r=tt;e=nt;var i=32-Fe(r)-1;r&=~(1<<i),n+=1;var o=32-Fe(t)+i;if(30<o){var s=i-i%5;o=(r&(1<<s)-1).toString(32),r>>=s,i-=s,tt=1<<32-Fe(t)+i|n<<i|r,nt=o+e}else tt=1<<o|n<<i|r,nt=e}function cl(e){e.return!==null&&(qt(e,1),_f(e,1,0))}function dl(e){for(;e===to;)to=Nn[--On],Nn[On]=null,no=Nn[--On],Nn[On]=null;for(;e===on;)on=Pe[--Re],Pe[Re]=null,nt=Pe[--Re],Pe[Re]=null,tt=Pe[--Re],Pe[Re]=null}var Ee=null,Se=null,B=!1,je=null;function Sf(e,t){var n=Ne(5,null,null,0);n.elementType="DELETED",n.stateNode=t,n.return=e,t=e.deletions,t===null?(e.deletions=[n],e.flags|=16):t.push(n)}function Bu(e,t){switch(e.tag){case 5:var n=e.type;return t=t.nodeType!==1||n.toLowerCase()!==t.nodeName.toLowerCase()?null:t,t!==null?(e.stateNode=t,Ee=e,Se=Lt(t.firstChild),!0):!1;case 6:return t=e.pendingProps===""||t.nodeType!==3?null:t,t!==null?(e.stateNode=t,Ee=e,Se=null,!0):!1;case 13:return t=t.nodeType!==8?null:t,t!==null?(n=on!==null?{id:tt,overflow:nt}:null,e.memoizedState={dehydrated:t,treeContext:n,retryLane:1073741824},n=Ne(18,null,null,0),n.stateNode=t,n.return=e,e.child=n,Ee=e,Se=null,!0):!1;default:return!1}}function fa(e){return(e.mode&1)!==0&&(e.flags&128)===0}function ha(e){if(B){var t=Se;if(t){var n=t;if(!Bu(e,t)){if(fa(e))throw Error(_(418));t=Lt(n.nextSibling);var r=Ee;t&&Bu(e,t)?Sf(r,n):(e.flags=e.flags&-4097|2,B=!1,Ee=e)}}else{if(fa(e))throw Error(_(418));e.flags=e.flags&-4097|2,B=!1,Ee=e}}}function Vu(e){for(e=e.return;e!==null&&e.tag!==5&&e.tag!==3&&e.tag!==13;)e=e.return;Ee=e}function Si(e){if(e!==Ee)return!1;if(!B)return Vu(e),B=!0,!1;var t;if((t=e.tag!==3)&&!(t=e.tag!==5)&&(t=e.type,t=t!=="head"&&t!=="body"&&!la(e.type,e.memoizedProps)),t&&(t=Se)){if(fa(e))throw Ef(),Error(_(418));for(;t;)Sf(e,t),t=Lt(t.nextSibling)}if(Vu(e),e.tag===13){if(e=e.memoizedState,e=e!==null?e.dehydrated:null,!e)throw Error(_(317));e:{for(e=e.nextSibling,t=0;e;){if(e.nodeType===8){var n=e.data;if(n==="/$"){if(t===0){Se=Lt(e.nextSibling);break e}t--}else n!=="$"&&n!=="$!"&&n!=="$?"||t++}e=e.nextSibling}Se=null}}else Se=Ee?Lt(e.stateNode.nextSibling):null;return!0}function Ef(){for(var e=Se;e;)e=Lt(e.nextSibling)}function Kn(){Se=Ee=null,B=!1}function fl(e){je===null?je=[e]:je.push(e)}var Jg=mt.ReactCurrentBatchConfig;function ur(e,t,n){if(e=n.ref,e!==null&&typeof e!="function"&&typeof e!="object"){if(n._owner){if(n=n._owner,n){if(n.tag!==1)throw Error(_(309));var r=n.stateNode}if(!r)throw Error(_(147,e));var i=r,o=""+e;return t!==null&&t.ref!==null&&typeof t.ref=="function"&&t.ref._stringRef===o?t.ref:(t=function(s){var a=i.refs;s===null?delete a[o]:a[o]=s},t._stringRef=o,t)}if(typeof e!="string")throw Error(_(284));if(!n._owner)throw Error(_(290,e))}return e}function Ei(e,t){throw e=Object.prototype.toString.call(t),Error(_(31,e==="[object Object]"?"object with keys {"+Object.keys(t).join(", ")+"}":e))}function Hu(e){var t=e._init;return t(e._payload)}function If(e){function t(h,c){if(e){var p=h.deletions;p===null?(h.deletions=[c],h.flags|=16):p.push(c)}}function n(h,c){if(!e)return null;for(;c!==null;)t(h,c),c=c.sibling;return null}function r(h,c){for(h=new Map;c!==null;)c.key!==null?h.set(c.key,c):h.set(c.index,c),c=c.sibling;return h}function i(h,c){return h=jt(h,c),h.index=0,h.sibling=null,h}function o(h,c,p){return h.index=p,e?(p=h.alternate,p!==null?(p=p.index,p<c?(h.flags|=2,c):p):(h.flags|=2,c)):(h.flags|=1048576,c)}function s(h){return e&&h.alternate===null&&(h.flags|=2),h}function a(h,c,p,v){return c===null||c.tag!==6?(c=Cs(p,h.mode,v),c.return=h,c):(c=i(c,p),c.return=h,c)}function l(h,c,p,v){var S=p.type;return S===In?d(h,c,p.props.children,v,p.key):c!==null&&(c.elementType===S||typeof S=="object"&&S!==null&&S.$$typeof===wt&&Hu(S)===c.type)?(v=i(c,p.props),v.ref=ur(h,c,p),v.return=h,v):(v=zi(p.type,p.key,p.props,null,h.mode,v),v.ref=ur(h,c,p),v.return=h,v)}function u(h,c,p,v){return c===null||c.tag!==4||c.stateNode.containerInfo!==p.containerInfo||c.stateNode.implementation!==p.implementation?(c=Ts(p,h.mode,v),c.return=h,c):(c=i(c,p.children||[]),c.return=h,c)}function d(h,c,p,v,S){return c===null||c.tag!==7?(c=nn(p,h.mode,v,S),c.return=h,c):(c=i(c,p),c.return=h,c)}function f(h,c,p){if(typeof c=="string"&&c!==""||typeof c=="number")return c=Cs(""+c,h.mode,p),c.return=h,c;if(typeof c=="object"&&c!==null){switch(c.$$typeof){case di:return p=zi(c.type,c.key,c.props,null,h.mode,p),p.ref=ur(h,null,c),p.return=h,p;case En:return c=Ts(c,h.mode,p),c.return=h,c;case wt:var v=c._init;return f(h,v(c._payload),p)}if(hr(c)||ir(c))return c=nn(c,h.mode,p,null),c.return=h,c;Ei(h,c)}return null}function m(h,c,p,v){var S=c!==null?c.key:null;if(typeof p=="string"&&p!==""||typeof p=="number")return S!==null?null:a(h,c,""+p,v);if(typeof p=="object"&&p!==null){switch(p.$$typeof){case di:return p.key===S?l(h,c,p,v):null;case En:return p.key===S?u(h,c,p,v):null;case wt:return S=p._init,m(h,c,S(p._payload),v)}if(hr(p)||ir(p))return S!==null?null:d(h,c,p,v,null);Ei(h,p)}return null}function g(h,c,p,v,S){if(typeof v=="string"&&v!==""||typeof v=="number")return h=h.get(p)||null,a(c,h,""+v,S);if(typeof v=="object"&&v!==null){switch(v.$$typeof){case di:return h=h.get(v.key===null?p:v.key)||null,l(c,h,v,S);case En:return h=h.get(v.key===null?p:v.key)||null,u(c,h,v,S);case wt:var P=v._init;return g(h,c,p,P(v._payload),S)}if(hr(v)||ir(v))return h=h.get(p)||null,d(c,h,v,S,null);Ei(c,v)}return null}function y(h,c,p,v){for(var S=null,P=null,R=c,N=c=0,z=null;R!==null&&N<p.length;N++){R.index>N?(z=R,R=null):z=R.sibling;var L=m(h,R,p[N],v);if(L===null){R===null&&(R=z);break}e&&R&&L.alternate===null&&t(h,R),c=o(L,c,N),P===null?S=L:P.sibling=L,P=L,R=z}if(N===p.length)return n(h,R),B&&qt(h,N),S;if(R===null){for(;N<p.length;N++)R=f(h,p[N],v),R!==null&&(c=o(R,c,N),P===null?S=R:P.sibling=R,P=R);return B&&qt(h,N),S}for(R=r(h,R);N<p.length;N++)z=g(R,h,N,p[N],v),z!==null&&(e&&z.alternate!==null&&R.delete(z.key===null?N:z.key),c=o(z,c,N),P===null?S=z:P.sibling=z,P=z);return e&&R.forEach(function(Le){return t(h,Le)}),B&&qt(h,N),S}function w(h,c,p,v){var S=ir(p);if(typeof S!="function")throw Error(_(150));if(p=S.call(p),p==null)throw Error(_(151));for(var P=S=null,R=c,N=c=0,z=null,L=p.next();R!==null&&!L.done;N++,L=p.next()){R.index>N?(z=R,R=null):z=R.sibling;var Le=m(h,R,L.value,v);if(Le===null){R===null&&(R=z);break}e&&R&&Le.alternate===null&&t(h,R),c=o(Le,c,N),P===null?S=Le:P.sibling=Le,P=Le,R=z}if(L.done)return n(h,R),B&&qt(h,N),S;if(R===null){for(;!L.done;N++,L=p.next())L=f(h,L.value,v),L!==null&&(c=o(L,c,N),P===null?S=L:P.sibling=L,P=L);return B&&qt(h,N),S}for(R=r(h,R);!L.done;N++,L=p.next())L=g(R,h,N,L.value,v),L!==null&&(e&&L.alternate!==null&&R.delete(L.key===null?N:L.key),c=o(L,c,N),P===null?S=L:P.sibling=L,P=L);return e&&R.forEach(function(nr){return t(h,nr)}),B&&qt(h,N),S}function E(h,c,p,v){if(typeof p=="object"&&p!==null&&p.type===In&&p.key===null&&(p=p.props.children),typeof p=="object"&&p!==null){switch(p.$$typeof){case di:e:{for(var S=p.key,P=c;P!==null;){if(P.key===S){if(S=p.type,S===In){if(P.tag===7){n(h,P.sibling),c=i(P,p.props.children),c.return=h,h=c;break e}}else if(P.elementType===S||typeof S=="object"&&S!==null&&S.$$typeof===wt&&Hu(S)===P.type){n(h,P.sibling),c=i(P,p.props),c.ref=ur(h,P,p),c.return=h,h=c;break e}n(h,P);break}else t(h,P);P=P.sibling}p.type===In?(c=nn(p.props.children,h.mode,v,p.key),c.return=h,h=c):(v=zi(p.type,p.key,p.props,null,h.mode,v),v.ref=ur(h,c,p),v.return=h,h=v)}return s(h);case En:e:{for(P=p.key;c!==null;){if(c.key===P)if(c.tag===4&&c.stateNode.containerInfo===p.containerInfo&&c.stateNode.implementation===p.implementation){n(h,c.sibling),c=i(c,p.children||[]),c.return=h,h=c;break e}else{n(h,c);break}else t(h,c);c=c.sibling}c=Ts(p,h.mode,v),c.return=h,h=c}return s(h);case wt:return P=p._init,E(h,c,P(p._payload),v)}if(hr(p))return y(h,c,p,v);if(ir(p))return w(h,c,p,v);Ei(h,p)}return typeof p=="string"&&p!==""||typeof p=="number"?(p=""+p,c!==null&&c.tag===6?(n(h,c.sibling),c=i(c,p),c.return=h,h=c):(n(h,c),c=Cs(p,h.mode,v),c.return=h,h=c),s(h)):n(h,c)}return E}var Gn=If(!0),kf=If(!1),ro=Ht(null),io=null,An=null,hl=null;function pl(){hl=An=io=null}function ml(e){var t=ro.current;$(ro),e._currentValue=t}function pa(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)!==t?(e.childLanes|=t,r!==null&&(r.childLanes|=t)):r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t),e===n)break;e=e.return}}function zn(e,t){io=e,hl=An=null,e=e.dependencies,e!==null&&e.firstContext!==null&&(e.lanes&t&&(me=!0),e.firstContext=null)}function Ae(e){var t=e._currentValue;if(hl!==e)if(e={context:e,memoizedValue:t,next:null},An===null){if(io===null)throw Error(_(308));An=e,io.dependencies={lanes:0,firstContext:e}}else An=An.next=e;return t}var Xt=null;function gl(e){Xt===null?Xt=[e]:Xt.push(e)}function Cf(e,t,n,r){var i=t.interleaved;return i===null?(n.next=n,gl(t)):(n.next=i.next,i.next=n),t.interleaved=n,dt(e,r)}function dt(e,t){e.lanes|=t;var n=e.alternate;for(n!==null&&(n.lanes|=t),n=e,e=e.return;e!==null;)e.childLanes|=t,n=e.alternate,n!==null&&(n.childLanes|=t),n=e,e=e.return;return n.tag===3?n.stateNode:null}var _t=!1;function vl(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,interleaved:null,lanes:0},effects:null}}function Tf(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,effects:e.effects})}function at(e,t){return{eventTime:e,lane:t,tag:0,payload:null,callback:null,next:null}}function Dt(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,M&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,dt(e,n)}return i=r.interleaved,i===null?(t.next=t,gl(r)):(t.next=i.next,i.next=t),r.interleaved=t,dt(e,n)}function Li(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,(n&4194240)!==0)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,nl(e,n)}}function Wu(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,o=null;if(n=n.firstBaseUpdate,n!==null){do{var s={eventTime:n.eventTime,lane:n.lane,tag:n.tag,payload:n.payload,callback:n.callback,next:null};o===null?i=o=s:o=o.next=s,n=n.next}while(n!==null);o===null?i=o=t:o=o.next=t}else i=o=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:o,shared:r.shared,effects:r.effects},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}function oo(e,t,n,r){var i=e.updateQueue;_t=!1;var o=i.firstBaseUpdate,s=i.lastBaseUpdate,a=i.shared.pending;if(a!==null){i.shared.pending=null;var l=a,u=l.next;l.next=null,s===null?o=u:s.next=u,s=l;var d=e.alternate;d!==null&&(d=d.updateQueue,a=d.lastBaseUpdate,a!==s&&(a===null?d.firstBaseUpdate=u:a.next=u,d.lastBaseUpdate=l))}if(o!==null){var f=i.baseState;s=0,d=u=l=null,a=o;do{var m=a.lane,g=a.eventTime;if((r&m)===m){d!==null&&(d=d.next={eventTime:g,lane:0,tag:a.tag,payload:a.payload,callback:a.callback,next:null});e:{var y=e,w=a;switch(m=t,g=n,w.tag){case 1:if(y=w.payload,typeof y=="function"){f=y.call(g,f,m);break e}f=y;break e;case 3:y.flags=y.flags&-65537|128;case 0:if(y=w.payload,m=typeof y=="function"?y.call(g,f,m):y,m==null)break e;f=W({},f,m);break e;case 2:_t=!0}}a.callback!==null&&a.lane!==0&&(e.flags|=64,m=i.effects,m===null?i.effects=[a]:m.push(a))}else g={eventTime:g,lane:m,tag:a.tag,payload:a.payload,callback:a.callback,next:null},d===null?(u=d=g,l=f):d=d.next=g,s|=m;if(a=a.next,a===null){if(a=i.shared.pending,a===null)break;m=a,a=m.next,m.next=null,i.lastBaseUpdate=m,i.shared.pending=null}}while(!0);if(d===null&&(l=f),i.baseState=l,i.firstBaseUpdate=u,i.lastBaseUpdate=d,t=i.shared.interleaved,t!==null){i=t;do s|=i.lane,i=i.next;while(i!==t)}else o===null&&(i.shared.lanes=0);an|=s,e.lanes=s,e.memoizedState=f}}function Ku(e,t,n){if(e=t.effects,t.effects=null,e!==null)for(t=0;t<e.length;t++){var r=e[t],i=r.callback;if(i!==null){if(r.callback=null,r=n,typeof i!="function")throw Error(_(191,i));i.call(r)}}}var ei={},Ke=Ht(ei),zr=Ht(ei),br=Ht(ei);function Zt(e){if(e===ei)throw Error(_(174));return e}function yl(e,t){switch(F(br,t),F(zr,e),F(Ke,ei),e=t.nodeType,e){case 9:case 11:t=(t=t.documentElement)?t.namespaceURI:qs(null,"");break;default:e=e===8?t.parentNode:t,t=e.namespaceURI||null,e=e.tagName,t=qs(t,e)}$(Ke),F(Ke,t)}function qn(){$(Ke),$(zr),$(br)}function Pf(e){Zt(br.current);var t=Zt(Ke.current),n=qs(t,e.type);t!==n&&(F(zr,e),F(Ke,n))}function wl(e){zr.current===e&&($(Ke),$(zr))}var V=Ht(0);function so(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||n.data==="$?"||n.data==="$!"))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==void 0){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var ws=[];function _l(){for(var e=0;e<ws.length;e++)ws[e]._workInProgressVersionPrimary=null;ws.length=0}var Di=mt.ReactCurrentDispatcher,_s=mt.ReactCurrentBatchConfig,sn=0,H=null,J=null,ee=null,ao=!1,Er=!1,$r=0,Xg=0;function oe(){throw Error(_(321))}function Sl(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!be(e[n],t[n]))return!1;return!0}function El(e,t,n,r,i,o){if(sn=o,H=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,Di.current=e===null||e.memoizedState===null?nv:rv,e=n(r,i),Er){o=0;do{if(Er=!1,$r=0,25<=o)throw Error(_(301));o+=1,ee=J=null,t.updateQueue=null,Di.current=iv,e=n(r,i)}while(Er)}if(Di.current=lo,t=J!==null&&J.next!==null,sn=0,ee=J=H=null,ao=!1,t)throw Error(_(300));return e}function Il(){var e=$r!==0;return $r=0,e}function Be(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return ee===null?H.memoizedState=ee=e:ee=ee.next=e,ee}function xe(){if(J===null){var e=H.alternate;e=e!==null?e.memoizedState:null}else e=J.next;var t=ee===null?H.memoizedState:ee.next;if(t!==null)ee=t,J=e;else{if(e===null)throw Error(_(310));J=e,e={memoizedState:J.memoizedState,baseState:J.baseState,baseQueue:J.baseQueue,queue:J.queue,next:null},ee===null?H.memoizedState=ee=e:ee=ee.next=e}return ee}function Br(e,t){return typeof t=="function"?t(e):t}function Ss(e){var t=xe(),n=t.queue;if(n===null)throw Error(_(311));n.lastRenderedReducer=e;var r=J,i=r.baseQueue,o=n.pending;if(o!==null){if(i!==null){var s=i.next;i.next=o.next,o.next=s}r.baseQueue=i=o,n.pending=null}if(i!==null){o=i.next,r=r.baseState;var a=s=null,l=null,u=o;do{var d=u.lane;if((sn&d)===d)l!==null&&(l=l.next={lane:0,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),r=u.hasEagerState?u.eagerState:e(r,u.action);else{var f={lane:d,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null};l===null?(a=l=f,s=r):l=l.next=f,H.lanes|=d,an|=d}u=u.next}while(u!==null&&u!==o);l===null?s=r:l.next=a,be(r,t.memoizedState)||(me=!0),t.memoizedState=r,t.baseState=s,t.baseQueue=l,n.lastRenderedState=r}if(e=n.interleaved,e!==null){i=e;do o=i.lane,H.lanes|=o,an|=o,i=i.next;while(i!==e)}else i===null&&(n.lanes=0);return[t.memoizedState,n.dispatch]}function Es(e){var t=xe(),n=t.queue;if(n===null)throw Error(_(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,o=t.memoizedState;if(i!==null){n.pending=null;var s=i=i.next;do o=e(o,s.action),s=s.next;while(s!==i);be(o,t.memoizedState)||(me=!0),t.memoizedState=o,t.baseQueue===null&&(t.baseState=o),n.lastRenderedState=o}return[o,r]}function Rf(){}function Nf(e,t){var n=H,r=xe(),i=t(),o=!be(r.memoizedState,i);if(o&&(r.memoizedState=i,me=!0),r=r.queue,kl(xf.bind(null,n,r,e),[e]),r.getSnapshot!==t||o||ee!==null&&ee.memoizedState.tag&1){if(n.flags|=2048,Vr(9,Af.bind(null,n,r,i,t),void 0,null),te===null)throw Error(_(349));sn&30||Of(n,t,i)}return i}function Of(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=H.updateQueue,t===null?(t={lastEffect:null,stores:null},H.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function Af(e,t,n,r){t.value=n,t.getSnapshot=r,Lf(t)&&Df(e)}function xf(e,t,n){return n(function(){Lf(t)&&Df(e)})}function Lf(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!be(e,n)}catch{return!0}}function Df(e){var t=dt(e,1);t!==null&&ze(t,e,1,-1)}function Gu(e){var t=Be();return typeof e=="function"&&(e=e()),t.memoizedState=t.baseState=e,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:Br,lastRenderedState:e},t.queue=e,e=e.dispatch=tv.bind(null,H,e),[t.memoizedState,e]}function Vr(e,t,n,r){return e={tag:e,create:t,destroy:n,deps:r,next:null},t=H.updateQueue,t===null?(t={lastEffect:null,stores:null},H.updateQueue=t,t.lastEffect=e.next=e):(n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e)),e}function Mf(){return xe().memoizedState}function Mi(e,t,n,r){var i=Be();H.flags|=e,i.memoizedState=Vr(1|t,n,void 0,r===void 0?null:r)}function Do(e,t,n,r){var i=xe();r=r===void 0?null:r;var o=void 0;if(J!==null){var s=J.memoizedState;if(o=s.destroy,r!==null&&Sl(r,s.deps)){i.memoizedState=Vr(t,n,o,r);return}}H.flags|=e,i.memoizedState=Vr(1|t,n,o,r)}function qu(e,t){return Mi(8390656,8,e,t)}function kl(e,t){return Do(2048,8,e,t)}function Uf(e,t){return Do(4,2,e,t)}function jf(e,t){return Do(4,4,e,t)}function Ff(e,t){if(typeof t=="function")return e=e(),t(e),function(){t(null)};if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function zf(e,t,n){return n=n!=null?n.concat([e]):null,Do(4,4,Ff.bind(null,t,e),n)}function Cl(){}function bf(e,t){var n=xe();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&Sl(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function $f(e,t){var n=xe();t=t===void 0?null:t;var r=n.memoizedState;return r!==null&&t!==null&&Sl(t,r[1])?r[0]:(e=e(),n.memoizedState=[e,t],e)}function Bf(e,t,n){return sn&21?(be(n,t)||(n=Kd(),H.lanes|=n,an|=n,e.baseState=!0),t):(e.baseState&&(e.baseState=!1,me=!0),e.memoizedState=n)}function Zg(e,t){var n=U;U=n!==0&&4>n?n:4,e(!0);var r=_s.transition;_s.transition={};try{e(!1),t()}finally{U=n,_s.transition=r}}function Vf(){return xe().memoizedState}function ev(e,t,n){var r=Ut(e);if(n={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null},Hf(e))Wf(t,n);else if(n=Cf(e,t,n,r),n!==null){var i=de();ze(n,e,r,i),Kf(n,t,r)}}function tv(e,t,n){var r=Ut(e),i={lane:r,action:n,hasEagerState:!1,eagerState:null,next:null};if(Hf(e))Wf(t,i);else{var o=e.alternate;if(e.lanes===0&&(o===null||o.lanes===0)&&(o=t.lastRenderedReducer,o!==null))try{var s=t.lastRenderedState,a=o(s,n);if(i.hasEagerState=!0,i.eagerState=a,be(a,s)){var l=t.interleaved;l===null?(i.next=i,gl(t)):(i.next=l.next,l.next=i),t.interleaved=i;return}}catch{}finally{}n=Cf(e,t,i,r),n!==null&&(i=de(),ze(n,e,r,i),Kf(n,t,r))}}function Hf(e){var t=e.alternate;return e===H||t!==null&&t===H}function Wf(e,t){Er=ao=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function Kf(e,t,n){if(n&4194240){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,nl(e,n)}}var lo={readContext:Ae,useCallback:oe,useContext:oe,useEffect:oe,useImperativeHandle:oe,useInsertionEffect:oe,useLayoutEffect:oe,useMemo:oe,useReducer:oe,useRef:oe,useState:oe,useDebugValue:oe,useDeferredValue:oe,useTransition:oe,useMutableSource:oe,useSyncExternalStore:oe,useId:oe,unstable_isNewReconciler:!1},nv={readContext:Ae,useCallback:function(e,t){return Be().memoizedState=[e,t===void 0?null:t],e},useContext:Ae,useEffect:qu,useImperativeHandle:function(e,t,n){return n=n!=null?n.concat([e]):null,Mi(4194308,4,Ff.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Mi(4194308,4,e,t)},useInsertionEffect:function(e,t){return Mi(4,2,e,t)},useMemo:function(e,t){var n=Be();return t=t===void 0?null:t,e=e(),n.memoizedState=[e,t],e},useReducer:function(e,t,n){var r=Be();return t=n!==void 0?n(t):t,r.memoizedState=r.baseState=t,e={pending:null,interleaved:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:t},r.queue=e,e=e.dispatch=ev.bind(null,H,e),[r.memoizedState,e]},useRef:function(e){var t=Be();return e={current:e},t.memoizedState=e},useState:Gu,useDebugValue:Cl,useDeferredValue:function(e){return Be().memoizedState=e},useTransition:function(){var e=Gu(!1),t=e[0];return e=Zg.bind(null,e[1]),Be().memoizedState=e,[t,e]},useMutableSource:function(){},useSyncExternalStore:function(e,t,n){var r=H,i=Be();if(B){if(n===void 0)throw Error(_(407));n=n()}else{if(n=t(),te===null)throw Error(_(349));sn&30||Of(r,t,n)}i.memoizedState=n;var o={value:n,getSnapshot:t};return i.queue=o,qu(xf.bind(null,r,o,e),[e]),r.flags|=2048,Vr(9,Af.bind(null,r,o,n,t),void 0,null),n},useId:function(){var e=Be(),t=te.identifierPrefix;if(B){var n=nt,r=tt;n=(r&~(1<<32-Fe(r)-1)).toString(32)+n,t=":"+t+"R"+n,n=$r++,0<n&&(t+="H"+n.toString(32)),t+=":"}else n=Xg++,t=":"+t+"r"+n.toString(32)+":";return e.memoizedState=t},unstable_isNewReconciler:!1},rv={readContext:Ae,useCallback:bf,useContext:Ae,useEffect:kl,useImperativeHandle:zf,useInsertionEffect:Uf,useLayoutEffect:jf,useMemo:$f,useReducer:Ss,useRef:Mf,useState:function(){return Ss(Br)},useDebugValue:Cl,useDeferredValue:function(e){var t=xe();return Bf(t,J.memoizedState,e)},useTransition:function(){var e=Ss(Br)[0],t=xe().memoizedState;return[e,t]},useMutableSource:Rf,useSyncExternalStore:Nf,useId:Vf,unstable_isNewReconciler:!1},iv={readContext:Ae,useCallback:bf,useContext:Ae,useEffect:kl,useImperativeHandle:zf,useInsertionEffect:Uf,useLayoutEffect:jf,useMemo:$f,useReducer:Es,useRef:Mf,useState:function(){return Es(Br)},useDebugValue:Cl,useDeferredValue:function(e){var t=xe();return J===null?t.memoizedState=e:Bf(t,J.memoizedState,e)},useTransition:function(){var e=Es(Br)[0],t=xe().memoizedState;return[e,t]},useMutableSource:Rf,useSyncExternalStore:Nf,useId:Vf,unstable_isNewReconciler:!1};function Me(e,t){if(e&&e.defaultProps){t=W({},t),e=e.defaultProps;for(var n in e)t[n]===void 0&&(t[n]=e[n]);return t}return t}function ma(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:W({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Mo={isMounted:function(e){return(e=e._reactInternals)?mn(e)===e:!1},enqueueSetState:function(e,t,n){e=e._reactInternals;var r=de(),i=Ut(e),o=at(r,i);o.payload=t,n!=null&&(o.callback=n),t=Dt(e,o,i),t!==null&&(ze(t,e,i,r),Li(t,e,i))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=de(),i=Ut(e),o=at(r,i);o.tag=1,o.payload=t,n!=null&&(o.callback=n),t=Dt(e,o,i),t!==null&&(ze(t,e,i,r),Li(t,e,i))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=de(),r=Ut(e),i=at(n,r);i.tag=2,t!=null&&(i.callback=t),t=Dt(e,i,r),t!==null&&(ze(t,e,r,n),Li(t,e,r))}};function Qu(e,t,n,r,i,o,s){return e=e.stateNode,typeof e.shouldComponentUpdate=="function"?e.shouldComponentUpdate(r,o,s):t.prototype&&t.prototype.isPureReactComponent?!Mr(n,r)||!Mr(i,o):!0}function Gf(e,t,n){var r=!1,i=Bt,o=t.contextType;return typeof o=="object"&&o!==null?o=Ae(o):(i=ve(t)?rn:le.current,r=t.contextTypes,o=(r=r!=null)?Wn(e,i):Bt),t=new t(n,o),e.memoizedState=t.state!==null&&t.state!==void 0?t.state:null,t.updater=Mo,e.stateNode=t,t._reactInternals=e,r&&(e=e.stateNode,e.__reactInternalMemoizedUnmaskedChildContext=i,e.__reactInternalMemoizedMaskedChildContext=o),t}function Yu(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps=="function"&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps=="function"&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&Mo.enqueueReplaceState(t,t.state,null)}function ga(e,t,n,r){var i=e.stateNode;i.props=n,i.state=e.memoizedState,i.refs={},vl(e);var o=t.contextType;typeof o=="object"&&o!==null?i.context=Ae(o):(o=ve(t)?rn:le.current,i.context=Wn(e,o)),i.state=e.memoizedState,o=t.getDerivedStateFromProps,typeof o=="function"&&(ma(e,t,o,n),i.state=e.memoizedState),typeof t.getDerivedStateFromProps=="function"||typeof i.getSnapshotBeforeUpdate=="function"||typeof i.UNSAFE_componentWillMount!="function"&&typeof i.componentWillMount!="function"||(t=i.state,typeof i.componentWillMount=="function"&&i.componentWillMount(),typeof i.UNSAFE_componentWillMount=="function"&&i.UNSAFE_componentWillMount(),t!==i.state&&Mo.enqueueReplaceState(i,i.state,null),oo(e,n,i,r),i.state=e.memoizedState),typeof i.componentDidMount=="function"&&(e.flags|=4194308)}function Qn(e,t){try{var n="",r=t;do n+=xm(r),r=r.return;while(r);var i=n}catch(o){i=`
Error generating stack: `+o.message+`
`+o.stack}return{value:e,source:t,stack:i,digest:null}}function Is(e,t,n){return{value:e,source:null,stack:n??null,digest:t??null}}function va(e,t){try{console.error(t.value)}catch(n){setTimeout(function(){throw n})}}var ov=typeof WeakMap=="function"?WeakMap:Map;function qf(e,t,n){n=at(-1,n),n.tag=3,n.payload={element:null};var r=t.value;return n.callback=function(){co||(co=!0,Pa=r),va(e,t)},n}function Qf(e,t,n){n=at(-1,n),n.tag=3;var r=e.type.getDerivedStateFromError;if(typeof r=="function"){var i=t.value;n.payload=function(){return r(i)},n.callback=function(){va(e,t)}}var o=e.stateNode;return o!==null&&typeof o.componentDidCatch=="function"&&(n.callback=function(){va(e,t),typeof r!="function"&&(Mt===null?Mt=new Set([this]):Mt.add(this));var s=t.stack;this.componentDidCatch(t.value,{componentStack:s!==null?s:""})}),n}function Ju(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new ov;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(i.add(n),e=wv.bind(null,e,t,n),t.then(e,e))}function Xu(e){do{var t;if((t=e.tag===13)&&(t=e.memoizedState,t=t!==null?t.dehydrated!==null:!0),t)return e;e=e.return}while(e!==null);return null}function Zu(e,t,n,r,i){return e.mode&1?(e.flags|=65536,e.lanes=i,e):(e===t?e.flags|=65536:(e.flags|=128,n.flags|=131072,n.flags&=-52805,n.tag===1&&(n.alternate===null?n.tag=17:(t=at(-1,1),t.tag=2,Dt(n,t,1))),n.lanes|=1),e)}var sv=mt.ReactCurrentOwner,me=!1;function ce(e,t,n,r){t.child=e===null?kf(t,null,n,r):Gn(t,e.child,n,r)}function ec(e,t,n,r,i){n=n.render;var o=t.ref;return zn(t,i),r=El(e,t,n,r,o,i),n=Il(),e!==null&&!me?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,ft(e,t,i)):(B&&n&&cl(t),t.flags|=1,ce(e,t,r,i),t.child)}function tc(e,t,n,r,i){if(e===null){var o=n.type;return typeof o=="function"&&!Ll(o)&&o.defaultProps===void 0&&n.compare===null&&n.defaultProps===void 0?(t.tag=15,t.type=o,Yf(e,t,o,r,i)):(e=zi(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(o=e.child,!(e.lanes&i)){var s=o.memoizedProps;if(n=n.compare,n=n!==null?n:Mr,n(s,r)&&e.ref===t.ref)return ft(e,t,i)}return t.flags|=1,e=jt(o,r),e.ref=t.ref,e.return=t,t.child=e}function Yf(e,t,n,r,i){if(e!==null){var o=e.memoizedProps;if(Mr(o,r)&&e.ref===t.ref)if(me=!1,t.pendingProps=r=o,(e.lanes&i)!==0)e.flags&131072&&(me=!0);else return t.lanes=e.lanes,ft(e,t,i)}return ya(e,t,n,r,i)}function Jf(e,t,n){var r=t.pendingProps,i=r.children,o=e!==null?e.memoizedState:null;if(r.mode==="hidden")if(!(t.mode&1))t.memoizedState={baseLanes:0,cachePool:null,transitions:null},F(Ln,_e),_e|=n;else{if(!(n&1073741824))return e=o!==null?o.baseLanes|n:n,t.lanes=t.childLanes=1073741824,t.memoizedState={baseLanes:e,cachePool:null,transitions:null},t.updateQueue=null,F(Ln,_e),_e|=e,null;t.memoizedState={baseLanes:0,cachePool:null,transitions:null},r=o!==null?o.baseLanes:n,F(Ln,_e),_e|=r}else o!==null?(r=o.baseLanes|n,t.memoizedState=null):r=n,F(Ln,_e),_e|=r;return ce(e,t,i,n),t.child}function Xf(e,t){var n=t.ref;(e===null&&n!==null||e!==null&&e.ref!==n)&&(t.flags|=512,t.flags|=2097152)}function ya(e,t,n,r,i){var o=ve(n)?rn:le.current;return o=Wn(t,o),zn(t,i),n=El(e,t,n,r,o,i),r=Il(),e!==null&&!me?(t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~i,ft(e,t,i)):(B&&r&&cl(t),t.flags|=1,ce(e,t,n,i),t.child)}function nc(e,t,n,r,i){if(ve(n)){var o=!0;eo(t)}else o=!1;if(zn(t,i),t.stateNode===null)Ui(e,t),Gf(t,n,r),ga(t,n,r,i),r=!0;else if(e===null){var s=t.stateNode,a=t.memoizedProps;s.props=a;var l=s.context,u=n.contextType;typeof u=="object"&&u!==null?u=Ae(u):(u=ve(n)?rn:le.current,u=Wn(t,u));var d=n.getDerivedStateFromProps,f=typeof d=="function"||typeof s.getSnapshotBeforeUpdate=="function";f||typeof s.UNSAFE_componentWillReceiveProps!="function"&&typeof s.componentWillReceiveProps!="function"||(a!==r||l!==u)&&Yu(t,s,r,u),_t=!1;var m=t.memoizedState;s.state=m,oo(t,r,s,i),l=t.memoizedState,a!==r||m!==l||ge.current||_t?(typeof d=="function"&&(ma(t,n,d,r),l=t.memoizedState),(a=_t||Qu(t,n,a,r,m,l,u))?(f||typeof s.UNSAFE_componentWillMount!="function"&&typeof s.componentWillMount!="function"||(typeof s.componentWillMount=="function"&&s.componentWillMount(),typeof s.UNSAFE_componentWillMount=="function"&&s.UNSAFE_componentWillMount()),typeof s.componentDidMount=="function"&&(t.flags|=4194308)):(typeof s.componentDidMount=="function"&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),s.props=r,s.state=l,s.context=u,r=a):(typeof s.componentDidMount=="function"&&(t.flags|=4194308),r=!1)}else{s=t.stateNode,Tf(e,t),a=t.memoizedProps,u=t.type===t.elementType?a:Me(t.type,a),s.props=u,f=t.pendingProps,m=s.context,l=n.contextType,typeof l=="object"&&l!==null?l=Ae(l):(l=ve(n)?rn:le.current,l=Wn(t,l));var g=n.getDerivedStateFromProps;(d=typeof g=="function"||typeof s.getSnapshotBeforeUpdate=="function")||typeof s.UNSAFE_componentWillReceiveProps!="function"&&typeof s.componentWillReceiveProps!="function"||(a!==f||m!==l)&&Yu(t,s,r,l),_t=!1,m=t.memoizedState,s.state=m,oo(t,r,s,i);var y=t.memoizedState;a!==f||m!==y||ge.current||_t?(typeof g=="function"&&(ma(t,n,g,r),y=t.memoizedState),(u=_t||Qu(t,n,u,r,m,y,l)||!1)?(d||typeof s.UNSAFE_componentWillUpdate!="function"&&typeof s.componentWillUpdate!="function"||(typeof s.componentWillUpdate=="function"&&s.componentWillUpdate(r,y,l),typeof s.UNSAFE_componentWillUpdate=="function"&&s.UNSAFE_componentWillUpdate(r,y,l)),typeof s.componentDidUpdate=="function"&&(t.flags|=4),typeof s.getSnapshotBeforeUpdate=="function"&&(t.flags|=1024)):(typeof s.componentDidUpdate!="function"||a===e.memoizedProps&&m===e.memoizedState||(t.flags|=4),typeof s.getSnapshotBeforeUpdate!="function"||a===e.memoizedProps&&m===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=y),s.props=r,s.state=y,s.context=l,r=u):(typeof s.componentDidUpdate!="function"||a===e.memoizedProps&&m===e.memoizedState||(t.flags|=4),typeof s.getSnapshotBeforeUpdate!="function"||a===e.memoizedProps&&m===e.memoizedState||(t.flags|=1024),r=!1)}return wa(e,t,n,r,o,i)}function wa(e,t,n,r,i,o){Xf(e,t);var s=(t.flags&128)!==0;if(!r&&!s)return i&&$u(t,n,!1),ft(e,t,o);r=t.stateNode,sv.current=t;var a=s&&typeof n.getDerivedStateFromError!="function"?null:r.render();return t.flags|=1,e!==null&&s?(t.child=Gn(t,e.child,null,o),t.child=Gn(t,null,a,o)):ce(e,t,a,o),t.memoizedState=r.state,i&&$u(t,n,!0),t.child}function Zf(e){var t=e.stateNode;t.pendingContext?bu(e,t.pendingContext,t.pendingContext!==t.context):t.context&&bu(e,t.context,!1),yl(e,t.containerInfo)}function rc(e,t,n,r,i){return Kn(),fl(i),t.flags|=256,ce(e,t,n,r),t.child}var _a={dehydrated:null,treeContext:null,retryLane:0};function Sa(e){return{baseLanes:e,cachePool:null,transitions:null}}function eh(e,t,n){var r=t.pendingProps,i=V.current,o=!1,s=(t.flags&128)!==0,a;if((a=s)||(a=e!==null&&e.memoizedState===null?!1:(i&2)!==0),a?(o=!0,t.flags&=-129):(e===null||e.memoizedState!==null)&&(i|=1),F(V,i&1),e===null)return ha(t),e=t.memoizedState,e!==null&&(e=e.dehydrated,e!==null)?(t.mode&1?e.data==="$!"?t.lanes=8:t.lanes=1073741824:t.lanes=1,null):(s=r.children,e=r.fallback,o?(r=t.mode,o=t.child,s={mode:"hidden",children:s},!(r&1)&&o!==null?(o.childLanes=0,o.pendingProps=s):o=Fo(s,r,0,null),e=nn(e,r,n,null),o.return=t,e.return=t,o.sibling=e,t.child=o,t.child.memoizedState=Sa(n),t.memoizedState=_a,e):Tl(t,s));if(i=e.memoizedState,i!==null&&(a=i.dehydrated,a!==null))return av(e,t,s,r,a,i,n);if(o){o=r.fallback,s=t.mode,i=e.child,a=i.sibling;var l={mode:"hidden",children:r.children};return!(s&1)&&t.child!==i?(r=t.child,r.childLanes=0,r.pendingProps=l,t.deletions=null):(r=jt(i,l),r.subtreeFlags=i.subtreeFlags&14680064),a!==null?o=jt(a,o):(o=nn(o,s,n,null),o.flags|=2),o.return=t,r.return=t,r.sibling=o,t.child=r,r=o,o=t.child,s=e.child.memoizedState,s=s===null?Sa(n):{baseLanes:s.baseLanes|n,cachePool:null,transitions:s.transitions},o.memoizedState=s,o.childLanes=e.childLanes&~n,t.memoizedState=_a,r}return o=e.child,e=o.sibling,r=jt(o,{mode:"visible",children:r.children}),!(t.mode&1)&&(r.lanes=n),r.return=t,r.sibling=null,e!==null&&(n=t.deletions,n===null?(t.deletions=[e],t.flags|=16):n.push(e)),t.child=r,t.memoizedState=null,r}function Tl(e,t){return t=Fo({mode:"visible",children:t},e.mode,0,null),t.return=e,e.child=t}function Ii(e,t,n,r){return r!==null&&fl(r),Gn(t,e.child,null,n),e=Tl(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function av(e,t,n,r,i,o,s){if(n)return t.flags&256?(t.flags&=-257,r=Is(Error(_(422))),Ii(e,t,s,r)):t.memoizedState!==null?(t.child=e.child,t.flags|=128,null):(o=r.fallback,i=t.mode,r=Fo({mode:"visible",children:r.children},i,0,null),o=nn(o,i,s,null),o.flags|=2,r.return=t,o.return=t,r.sibling=o,t.child=r,t.mode&1&&Gn(t,e.child,null,s),t.child.memoizedState=Sa(s),t.memoizedState=_a,o);if(!(t.mode&1))return Ii(e,t,s,null);if(i.data==="$!"){if(r=i.nextSibling&&i.nextSibling.dataset,r)var a=r.dgst;return r=a,o=Error(_(419)),r=Is(o,r,void 0),Ii(e,t,s,r)}if(a=(s&e.childLanes)!==0,me||a){if(r=te,r!==null){switch(s&-s){case 4:i=2;break;case 16:i=8;break;case 64:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:case 67108864:i=32;break;case 536870912:i=268435456;break;default:i=0}i=i&(r.suspendedLanes|s)?0:i,i!==0&&i!==o.retryLane&&(o.retryLane=i,dt(e,i),ze(r,e,i,-1))}return xl(),r=Is(Error(_(421))),Ii(e,t,s,r)}return i.data==="$?"?(t.flags|=128,t.child=e.child,t=_v.bind(null,e),i._reactRetry=t,null):(e=o.treeContext,Se=Lt(i.nextSibling),Ee=t,B=!0,je=null,e!==null&&(Pe[Re++]=tt,Pe[Re++]=nt,Pe[Re++]=on,tt=e.id,nt=e.overflow,on=t),t=Tl(t,r.children),t.flags|=4096,t)}function ic(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),pa(e.return,t,n)}function ks(e,t,n,r,i){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i)}function th(e,t,n){var r=t.pendingProps,i=r.revealOrder,o=r.tail;if(ce(e,t,r.children,n),r=V.current,r&2)r=r&1|2,t.flags|=128;else{if(e!==null&&e.flags&128)e:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&ic(e,n,t);else if(e.tag===19)ic(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break e;for(;e.sibling===null;){if(e.return===null||e.return===t)break e;e=e.return}e.sibling.return=e.return,e=e.sibling}r&=1}if(F(V,r),!(t.mode&1))t.memoizedState=null;else switch(i){case"forwards":for(n=t.child,i=null;n!==null;)e=n.alternate,e!==null&&so(e)===null&&(i=n),n=n.sibling;n=i,n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),ks(t,!1,i,n,o);break;case"backwards":for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&so(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}ks(t,!0,n,null,o);break;case"together":ks(t,!1,null,null,void 0);break;default:t.memoizedState=null}return t.child}function Ui(e,t){!(t.mode&1)&&e!==null&&(e.alternate=null,t.alternate=null,t.flags|=2)}function ft(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),an|=t.lanes,!(n&t.childLanes))return null;if(e!==null&&t.child!==e.child)throw Error(_(153));if(t.child!==null){for(e=t.child,n=jt(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=jt(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function lv(e,t,n){switch(t.tag){case 3:Zf(t),Kn();break;case 5:Pf(t);break;case 1:ve(t.type)&&eo(t);break;case 4:yl(t,t.stateNode.containerInfo);break;case 10:var r=t.type._context,i=t.memoizedProps.value;F(ro,r._currentValue),r._currentValue=i;break;case 13:if(r=t.memoizedState,r!==null)return r.dehydrated!==null?(F(V,V.current&1),t.flags|=128,null):n&t.child.childLanes?eh(e,t,n):(F(V,V.current&1),e=ft(e,t,n),e!==null?e.sibling:null);F(V,V.current&1);break;case 19:if(r=(n&t.childLanes)!==0,e.flags&128){if(r)return th(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),F(V,V.current),r)break;return null;case 22:case 23:return t.lanes=0,Jf(e,t,n)}return ft(e,t,n)}var nh,Ea,rh,ih;nh=function(e,t){for(var n=t.child;n!==null;){if(n.tag===5||n.tag===6)e.appendChild(n.stateNode);else if(n.tag!==4&&n.child!==null){n.child.return=n,n=n.child;continue}if(n===t)break;for(;n.sibling===null;){if(n.return===null||n.return===t)return;n=n.return}n.sibling.return=n.return,n=n.sibling}};Ea=function(){};rh=function(e,t,n,r){var i=e.memoizedProps;if(i!==r){e=t.stateNode,Zt(Ke.current);var o=null;switch(n){case"input":i=Hs(e,i),r=Hs(e,r),o=[];break;case"select":i=W({},i,{value:void 0}),r=W({},r,{value:void 0}),o=[];break;case"textarea":i=Gs(e,i),r=Gs(e,r),o=[];break;default:typeof i.onClick!="function"&&typeof r.onClick=="function"&&(e.onclick=Xi)}Qs(n,r);var s;n=null;for(u in i)if(!r.hasOwnProperty(u)&&i.hasOwnProperty(u)&&i[u]!=null)if(u==="style"){var a=i[u];for(s in a)a.hasOwnProperty(s)&&(n||(n={}),n[s]="")}else u!=="dangerouslySetInnerHTML"&&u!=="children"&&u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&u!=="autoFocus"&&(Rr.hasOwnProperty(u)?o||(o=[]):(o=o||[]).push(u,null));for(u in r){var l=r[u];if(a=i!=null?i[u]:void 0,r.hasOwnProperty(u)&&l!==a&&(l!=null||a!=null))if(u==="style")if(a){for(s in a)!a.hasOwnProperty(s)||l&&l.hasOwnProperty(s)||(n||(n={}),n[s]="");for(s in l)l.hasOwnProperty(s)&&a[s]!==l[s]&&(n||(n={}),n[s]=l[s])}else n||(o||(o=[]),o.push(u,n)),n=l;else u==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,a=a?a.__html:void 0,l!=null&&a!==l&&(o=o||[]).push(u,l)):u==="children"?typeof l!="string"&&typeof l!="number"||(o=o||[]).push(u,""+l):u!=="suppressContentEditableWarning"&&u!=="suppressHydrationWarning"&&(Rr.hasOwnProperty(u)?(l!=null&&u==="onScroll"&&b("scroll",e),o||a===l||(o=[])):(o=o||[]).push(u,l))}n&&(o=o||[]).push("style",n);var u=o;(t.updateQueue=u)&&(t.flags|=4)}};ih=function(e,t,n,r){n!==r&&(t.flags|=4)};function cr(e,t){if(!B)switch(e.tailMode){case"hidden":t=e.tail;for(var n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null;break;case"collapsed":n=e.tail;for(var r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null}}function se(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&14680064,r|=i.flags&14680064,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function uv(e,t,n){var r=t.pendingProps;switch(dl(t),t.tag){case 2:case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return se(t),null;case 1:return ve(t.type)&&Zi(),se(t),null;case 3:return r=t.stateNode,qn(),$(ge),$(le),_l(),r.pendingContext&&(r.context=r.pendingContext,r.pendingContext=null),(e===null||e.child===null)&&(Si(t)?t.flags|=4:e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,je!==null&&(Oa(je),je=null))),Ea(e,t),se(t),null;case 5:wl(t);var i=Zt(br.current);if(n=t.type,e!==null&&t.stateNode!=null)rh(e,t,n,r,i),e.ref!==t.ref&&(t.flags|=512,t.flags|=2097152);else{if(!r){if(t.stateNode===null)throw Error(_(166));return se(t),null}if(e=Zt(Ke.current),Si(t)){r=t.stateNode,n=t.type;var o=t.memoizedProps;switch(r[Ve]=t,r[Fr]=o,e=(t.mode&1)!==0,n){case"dialog":b("cancel",r),b("close",r);break;case"iframe":case"object":case"embed":b("load",r);break;case"video":case"audio":for(i=0;i<mr.length;i++)b(mr[i],r);break;case"source":b("error",r);break;case"img":case"image":case"link":b("error",r),b("load",r);break;case"details":b("toggle",r);break;case"input":hu(r,o),b("invalid",r);break;case"select":r._wrapperState={wasMultiple:!!o.multiple},b("invalid",r);break;case"textarea":mu(r,o),b("invalid",r)}Qs(n,o),i=null;for(var s in o)if(o.hasOwnProperty(s)){var a=o[s];s==="children"?typeof a=="string"?r.textContent!==a&&(o.suppressHydrationWarning!==!0&&_i(r.textContent,a,e),i=["children",a]):typeof a=="number"&&r.textContent!==""+a&&(o.suppressHydrationWarning!==!0&&_i(r.textContent,a,e),i=["children",""+a]):Rr.hasOwnProperty(s)&&a!=null&&s==="onScroll"&&b("scroll",r)}switch(n){case"input":fi(r),pu(r,o,!0);break;case"textarea":fi(r),gu(r);break;case"select":case"option":break;default:typeof o.onClick=="function"&&(r.onclick=Xi)}r=i,t.updateQueue=r,r!==null&&(t.flags|=4)}else{s=i.nodeType===9?i:i.ownerDocument,e==="http://www.w3.org/1999/xhtml"&&(e=Ad(n)),e==="http://www.w3.org/1999/xhtml"?n==="script"?(e=s.createElement("div"),e.innerHTML="<script><\/script>",e=e.removeChild(e.firstChild)):typeof r.is=="string"?e=s.createElement(n,{is:r.is}):(e=s.createElement(n),n==="select"&&(s=e,r.multiple?s.multiple=!0:r.size&&(s.size=r.size))):e=s.createElementNS(e,n),e[Ve]=t,e[Fr]=r,nh(e,t,!1,!1),t.stateNode=e;e:{switch(s=Ys(n,r),n){case"dialog":b("cancel",e),b("close",e),i=r;break;case"iframe":case"object":case"embed":b("load",e),i=r;break;case"video":case"audio":for(i=0;i<mr.length;i++)b(mr[i],e);i=r;break;case"source":b("error",e),i=r;break;case"img":case"image":case"link":b("error",e),b("load",e),i=r;break;case"details":b("toggle",e),i=r;break;case"input":hu(e,r),i=Hs(e,r),b("invalid",e);break;case"option":i=r;break;case"select":e._wrapperState={wasMultiple:!!r.multiple},i=W({},r,{value:void 0}),b("invalid",e);break;case"textarea":mu(e,r),i=Gs(e,r),b("invalid",e);break;default:i=r}Qs(n,i),a=i;for(o in a)if(a.hasOwnProperty(o)){var l=a[o];o==="style"?Dd(e,l):o==="dangerouslySetInnerHTML"?(l=l?l.__html:void 0,l!=null&&xd(e,l)):o==="children"?typeof l=="string"?(n!=="textarea"||l!=="")&&Nr(e,l):typeof l=="number"&&Nr(e,""+l):o!=="suppressContentEditableWarning"&&o!=="suppressHydrationWarning"&&o!=="autoFocus"&&(Rr.hasOwnProperty(o)?l!=null&&o==="onScroll"&&b("scroll",e):l!=null&&Ya(e,o,l,s))}switch(n){case"input":fi(e),pu(e,r,!1);break;case"textarea":fi(e),gu(e);break;case"option":r.value!=null&&e.setAttribute("value",""+$t(r.value));break;case"select":e.multiple=!!r.multiple,o=r.value,o!=null?Mn(e,!!r.multiple,o,!1):r.defaultValue!=null&&Mn(e,!!r.multiple,r.defaultValue,!0);break;default:typeof i.onClick=="function"&&(e.onclick=Xi)}switch(n){case"button":case"input":case"select":case"textarea":r=!!r.autoFocus;break e;case"img":r=!0;break e;default:r=!1}}r&&(t.flags|=4)}t.ref!==null&&(t.flags|=512,t.flags|=2097152)}return se(t),null;case 6:if(e&&t.stateNode!=null)ih(e,t,e.memoizedProps,r);else{if(typeof r!="string"&&t.stateNode===null)throw Error(_(166));if(n=Zt(br.current),Zt(Ke.current),Si(t)){if(r=t.stateNode,n=t.memoizedProps,r[Ve]=t,(o=r.nodeValue!==n)&&(e=Ee,e!==null))switch(e.tag){case 3:_i(r.nodeValue,n,(e.mode&1)!==0);break;case 5:e.memoizedProps.suppressHydrationWarning!==!0&&_i(r.nodeValue,n,(e.mode&1)!==0)}o&&(t.flags|=4)}else r=(n.nodeType===9?n:n.ownerDocument).createTextNode(r),r[Ve]=t,t.stateNode=r}return se(t),null;case 13:if($(V),r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(B&&Se!==null&&t.mode&1&&!(t.flags&128))Ef(),Kn(),t.flags|=98560,o=!1;else if(o=Si(t),r!==null&&r.dehydrated!==null){if(e===null){if(!o)throw Error(_(318));if(o=t.memoizedState,o=o!==null?o.dehydrated:null,!o)throw Error(_(317));o[Ve]=t}else Kn(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;se(t),o=!1}else je!==null&&(Oa(je),je=null),o=!0;if(!o)return t.flags&65536?t:null}return t.flags&128?(t.lanes=n,t):(r=r!==null,r!==(e!==null&&e.memoizedState!==null)&&r&&(t.child.flags|=8192,t.mode&1&&(e===null||V.current&1?X===0&&(X=3):xl())),t.updateQueue!==null&&(t.flags|=4),se(t),null);case 4:return qn(),Ea(e,t),e===null&&Ur(t.stateNode.containerInfo),se(t),null;case 10:return ml(t.type._context),se(t),null;case 17:return ve(t.type)&&Zi(),se(t),null;case 19:if($(V),o=t.memoizedState,o===null)return se(t),null;if(r=(t.flags&128)!==0,s=o.rendering,s===null)if(r)cr(o,!1);else{if(X!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(s=so(e),s!==null){for(t.flags|=128,cr(o,!1),r=s.updateQueue,r!==null&&(t.updateQueue=r,t.flags|=4),t.subtreeFlags=0,r=n,n=t.child;n!==null;)o=n,e=r,o.flags&=14680066,s=o.alternate,s===null?(o.childLanes=0,o.lanes=e,o.child=null,o.subtreeFlags=0,o.memoizedProps=null,o.memoizedState=null,o.updateQueue=null,o.dependencies=null,o.stateNode=null):(o.childLanes=s.childLanes,o.lanes=s.lanes,o.child=s.child,o.subtreeFlags=0,o.deletions=null,o.memoizedProps=s.memoizedProps,o.memoizedState=s.memoizedState,o.updateQueue=s.updateQueue,o.type=s.type,e=s.dependencies,o.dependencies=e===null?null:{lanes:e.lanes,firstContext:e.firstContext}),n=n.sibling;return F(V,V.current&1|2),t.child}e=e.sibling}o.tail!==null&&q()>Yn&&(t.flags|=128,r=!0,cr(o,!1),t.lanes=4194304)}else{if(!r)if(e=so(s),e!==null){if(t.flags|=128,r=!0,n=e.updateQueue,n!==null&&(t.updateQueue=n,t.flags|=4),cr(o,!0),o.tail===null&&o.tailMode==="hidden"&&!s.alternate&&!B)return se(t),null}else 2*q()-o.renderingStartTime>Yn&&n!==1073741824&&(t.flags|=128,r=!0,cr(o,!1),t.lanes=4194304);o.isBackwards?(s.sibling=t.child,t.child=s):(n=o.last,n!==null?n.sibling=s:t.child=s,o.last=s)}return o.tail!==null?(t=o.tail,o.rendering=t,o.tail=t.sibling,o.renderingStartTime=q(),t.sibling=null,n=V.current,F(V,r?n&1|2:n&1),t):(se(t),null);case 22:case 23:return Al(),r=t.memoizedState!==null,e!==null&&e.memoizedState!==null!==r&&(t.flags|=8192),r&&t.mode&1?_e&1073741824&&(se(t),t.subtreeFlags&6&&(t.flags|=8192)):se(t),null;case 24:return null;case 25:return null}throw Error(_(156,t.tag))}function cv(e,t){switch(dl(t),t.tag){case 1:return ve(t.type)&&Zi(),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return qn(),$(ge),$(le),_l(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 5:return wl(t),null;case 13:if($(V),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(_(340));Kn()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return $(V),null;case 4:return qn(),null;case 10:return ml(t.type._context),null;case 22:case 23:return Al(),null;case 24:return null;default:return null}}var ki=!1,ae=!1,dv=typeof WeakSet=="function"?WeakSet:Set,C=null;function xn(e,t){var n=e.ref;if(n!==null)if(typeof n=="function")try{n(null)}catch(r){K(e,t,r)}else n.current=null}function Ia(e,t,n){try{n()}catch(r){K(e,t,r)}}var oc=!1;function fv(e,t){if(sa=Qi,e=uf(),ul(e)){if("selectionStart"in e)var n={start:e.selectionStart,end:e.selectionEnd};else e:{n=(n=e.ownerDocument)&&n.defaultView||window;var r=n.getSelection&&n.getSelection();if(r&&r.rangeCount!==0){n=r.anchorNode;var i=r.anchorOffset,o=r.focusNode;r=r.focusOffset;try{n.nodeType,o.nodeType}catch{n=null;break e}var s=0,a=-1,l=-1,u=0,d=0,f=e,m=null;t:for(;;){for(var g;f!==n||i!==0&&f.nodeType!==3||(a=s+i),f!==o||r!==0&&f.nodeType!==3||(l=s+r),f.nodeType===3&&(s+=f.nodeValue.length),(g=f.firstChild)!==null;)m=f,f=g;for(;;){if(f===e)break t;if(m===n&&++u===i&&(a=s),m===o&&++d===r&&(l=s),(g=f.nextSibling)!==null)break;f=m,m=f.parentNode}f=g}n=a===-1||l===-1?null:{start:a,end:l}}else n=null}n=n||{start:0,end:0}}else n=null;for(aa={focusedElem:e,selectionRange:n},Qi=!1,C=t;C!==null;)if(t=C,e=t.child,(t.subtreeFlags&1028)!==0&&e!==null)e.return=t,C=e;else for(;C!==null;){t=C;try{var y=t.alternate;if(t.flags&1024)switch(t.tag){case 0:case 11:case 15:break;case 1:if(y!==null){var w=y.memoizedProps,E=y.memoizedState,h=t.stateNode,c=h.getSnapshotBeforeUpdate(t.elementType===t.type?w:Me(t.type,w),E);h.__reactInternalSnapshotBeforeUpdate=c}break;case 3:var p=t.stateNode.containerInfo;p.nodeType===1?p.textContent="":p.nodeType===9&&p.documentElement&&p.removeChild(p.documentElement);break;case 5:case 6:case 4:case 17:break;default:throw Error(_(163))}}catch(v){K(t,t.return,v)}if(e=t.sibling,e!==null){e.return=t.return,C=e;break}C=t.return}return y=oc,oc=!1,y}function Ir(e,t,n){var r=t.updateQueue;if(r=r!==null?r.lastEffect:null,r!==null){var i=r=r.next;do{if((i.tag&e)===e){var o=i.destroy;i.destroy=void 0,o!==void 0&&Ia(t,n,o)}i=i.next}while(i!==r)}}function Uo(e,t){if(t=t.updateQueue,t=t!==null?t.lastEffect:null,t!==null){var n=t=t.next;do{if((n.tag&e)===e){var r=n.create;n.destroy=r()}n=n.next}while(n!==t)}}function ka(e){var t=e.ref;if(t!==null){var n=e.stateNode;switch(e.tag){case 5:e=n;break;default:e=n}typeof t=="function"?t(e):t.current=e}}function oh(e){var t=e.alternate;t!==null&&(e.alternate=null,oh(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&(delete t[Ve],delete t[Fr],delete t[ca],delete t[qg],delete t[Qg])),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}function sh(e){return e.tag===5||e.tag===3||e.tag===4}function sc(e){e:for(;;){for(;e.sibling===null;){if(e.return===null||sh(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.flags&2||e.child===null||e.tag===4)continue e;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function Ca(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.nodeType===8?n.parentNode.insertBefore(e,t):n.insertBefore(e,t):(n.nodeType===8?(t=n.parentNode,t.insertBefore(e,n)):(t=n,t.appendChild(e)),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=Xi));else if(r!==4&&(e=e.child,e!==null))for(Ca(e,t,n),e=e.sibling;e!==null;)Ca(e,t,n),e=e.sibling}function Ta(e,t,n){var r=e.tag;if(r===5||r===6)e=e.stateNode,t?n.insertBefore(e,t):n.appendChild(e);else if(r!==4&&(e=e.child,e!==null))for(Ta(e,t,n),e=e.sibling;e!==null;)Ta(e,t,n),e=e.sibling}var ne=null,Ue=!1;function vt(e,t,n){for(n=n.child;n!==null;)ah(e,t,n),n=n.sibling}function ah(e,t,n){if(We&&typeof We.onCommitFiberUnmount=="function")try{We.onCommitFiberUnmount(Ro,n)}catch{}switch(n.tag){case 5:ae||xn(n,t);case 6:var r=ne,i=Ue;ne=null,vt(e,t,n),ne=r,Ue=i,ne!==null&&(Ue?(e=ne,n=n.stateNode,e.nodeType===8?e.parentNode.removeChild(n):e.removeChild(n)):ne.removeChild(n.stateNode));break;case 18:ne!==null&&(Ue?(e=ne,n=n.stateNode,e.nodeType===8?vs(e.parentNode,n):e.nodeType===1&&vs(e,n),Lr(e)):vs(ne,n.stateNode));break;case 4:r=ne,i=Ue,ne=n.stateNode.containerInfo,Ue=!0,vt(e,t,n),ne=r,Ue=i;break;case 0:case 11:case 14:case 15:if(!ae&&(r=n.updateQueue,r!==null&&(r=r.lastEffect,r!==null))){i=r=r.next;do{var o=i,s=o.destroy;o=o.tag,s!==void 0&&(o&2||o&4)&&Ia(n,t,s),i=i.next}while(i!==r)}vt(e,t,n);break;case 1:if(!ae&&(xn(n,t),r=n.stateNode,typeof r.componentWillUnmount=="function"))try{r.props=n.memoizedProps,r.state=n.memoizedState,r.componentWillUnmount()}catch(a){K(n,t,a)}vt(e,t,n);break;case 21:vt(e,t,n);break;case 22:n.mode&1?(ae=(r=ae)||n.memoizedState!==null,vt(e,t,n),ae=r):vt(e,t,n);break;default:vt(e,t,n)}}function ac(e){var t=e.updateQueue;if(t!==null){e.updateQueue=null;var n=e.stateNode;n===null&&(n=e.stateNode=new dv),t.forEach(function(r){var i=Sv.bind(null,e,r);n.has(r)||(n.add(r),r.then(i,i))})}}function De(e,t){var n=t.deletions;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];try{var o=e,s=t,a=s;e:for(;a!==null;){switch(a.tag){case 5:ne=a.stateNode,Ue=!1;break e;case 3:ne=a.stateNode.containerInfo,Ue=!0;break e;case 4:ne=a.stateNode.containerInfo,Ue=!0;break e}a=a.return}if(ne===null)throw Error(_(160));ah(o,s,i),ne=null,Ue=!1;var l=i.alternate;l!==null&&(l.return=null),i.return=null}catch(u){K(i,t,u)}}if(t.subtreeFlags&12854)for(t=t.child;t!==null;)lh(t,e),t=t.sibling}function lh(e,t){var n=e.alternate,r=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(De(t,e),$e(e),r&4){try{Ir(3,e,e.return),Uo(3,e)}catch(w){K(e,e.return,w)}try{Ir(5,e,e.return)}catch(w){K(e,e.return,w)}}break;case 1:De(t,e),$e(e),r&512&&n!==null&&xn(n,n.return);break;case 5:if(De(t,e),$e(e),r&512&&n!==null&&xn(n,n.return),e.flags&32){var i=e.stateNode;try{Nr(i,"")}catch(w){K(e,e.return,w)}}if(r&4&&(i=e.stateNode,i!=null)){var o=e.memoizedProps,s=n!==null?n.memoizedProps:o,a=e.type,l=e.updateQueue;if(e.updateQueue=null,l!==null)try{a==="input"&&o.type==="radio"&&o.name!=null&&Nd(i,o),Ys(a,s);var u=Ys(a,o);for(s=0;s<l.length;s+=2){var d=l[s],f=l[s+1];d==="style"?Dd(i,f):d==="dangerouslySetInnerHTML"?xd(i,f):d==="children"?Nr(i,f):Ya(i,d,f,u)}switch(a){case"input":Ws(i,o);break;case"textarea":Od(i,o);break;case"select":var m=i._wrapperState.wasMultiple;i._wrapperState.wasMultiple=!!o.multiple;var g=o.value;g!=null?Mn(i,!!o.multiple,g,!1):m!==!!o.multiple&&(o.defaultValue!=null?Mn(i,!!o.multiple,o.defaultValue,!0):Mn(i,!!o.multiple,o.multiple?[]:"",!1))}i[Fr]=o}catch(w){K(e,e.return,w)}}break;case 6:if(De(t,e),$e(e),r&4){if(e.stateNode===null)throw Error(_(162));i=e.stateNode,o=e.memoizedProps;try{i.nodeValue=o}catch(w){K(e,e.return,w)}}break;case 3:if(De(t,e),$e(e),r&4&&n!==null&&n.memoizedState.isDehydrated)try{Lr(t.containerInfo)}catch(w){K(e,e.return,w)}break;case 4:De(t,e),$e(e);break;case 13:De(t,e),$e(e),i=e.child,i.flags&8192&&(o=i.memoizedState!==null,i.stateNode.isHidden=o,!o||i.alternate!==null&&i.alternate.memoizedState!==null||(Nl=q())),r&4&&ac(e);break;case 22:if(d=n!==null&&n.memoizedState!==null,e.mode&1?(ae=(u=ae)||d,De(t,e),ae=u):De(t,e),$e(e),r&8192){if(u=e.memoizedState!==null,(e.stateNode.isHidden=u)&&!d&&e.mode&1)for(C=e,d=e.child;d!==null;){for(f=C=d;C!==null;){switch(m=C,g=m.child,m.tag){case 0:case 11:case 14:case 15:Ir(4,m,m.return);break;case 1:xn(m,m.return);var y=m.stateNode;if(typeof y.componentWillUnmount=="function"){r=m,n=m.return;try{t=r,y.props=t.memoizedProps,y.state=t.memoizedState,y.componentWillUnmount()}catch(w){K(r,n,w)}}break;case 5:xn(m,m.return);break;case 22:if(m.memoizedState!==null){uc(f);continue}}g!==null?(g.return=m,C=g):uc(f)}d=d.sibling}e:for(d=null,f=e;;){if(f.tag===5){if(d===null){d=f;try{i=f.stateNode,u?(o=i.style,typeof o.setProperty=="function"?o.setProperty("display","none","important"):o.display="none"):(a=f.stateNode,l=f.memoizedProps.style,s=l!=null&&l.hasOwnProperty("display")?l.display:null,a.style.display=Ld("display",s))}catch(w){K(e,e.return,w)}}}else if(f.tag===6){if(d===null)try{f.stateNode.nodeValue=u?"":f.memoizedProps}catch(w){K(e,e.return,w)}}else if((f.tag!==22&&f.tag!==23||f.memoizedState===null||f===e)&&f.child!==null){f.child.return=f,f=f.child;continue}if(f===e)break e;for(;f.sibling===null;){if(f.return===null||f.return===e)break e;d===f&&(d=null),f=f.return}d===f&&(d=null),f.sibling.return=f.return,f=f.sibling}}break;case 19:De(t,e),$e(e),r&4&&ac(e);break;case 21:break;default:De(t,e),$e(e)}}function $e(e){var t=e.flags;if(t&2){try{e:{for(var n=e.return;n!==null;){if(sh(n)){var r=n;break e}n=n.return}throw Error(_(160))}switch(r.tag){case 5:var i=r.stateNode;r.flags&32&&(Nr(i,""),r.flags&=-33);var o=sc(e);Ta(e,o,i);break;case 3:case 4:var s=r.stateNode.containerInfo,a=sc(e);Ca(e,a,s);break;default:throw Error(_(161))}}catch(l){K(e,e.return,l)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function hv(e,t,n){C=e,uh(e)}function uh(e,t,n){for(var r=(e.mode&1)!==0;C!==null;){var i=C,o=i.child;if(i.tag===22&&r){var s=i.memoizedState!==null||ki;if(!s){var a=i.alternate,l=a!==null&&a.memoizedState!==null||ae;a=ki;var u=ae;if(ki=s,(ae=l)&&!u)for(C=i;C!==null;)s=C,l=s.child,s.tag===22&&s.memoizedState!==null?cc(i):l!==null?(l.return=s,C=l):cc(i);for(;o!==null;)C=o,uh(o),o=o.sibling;C=i,ki=a,ae=u}lc(e)}else i.subtreeFlags&8772&&o!==null?(o.return=i,C=o):lc(e)}}function lc(e){for(;C!==null;){var t=C;if(t.flags&8772){var n=t.alternate;try{if(t.flags&8772)switch(t.tag){case 0:case 11:case 15:ae||Uo(5,t);break;case 1:var r=t.stateNode;if(t.flags&4&&!ae)if(n===null)r.componentDidMount();else{var i=t.elementType===t.type?n.memoizedProps:Me(t.type,n.memoizedProps);r.componentDidUpdate(i,n.memoizedState,r.__reactInternalSnapshotBeforeUpdate)}var o=t.updateQueue;o!==null&&Ku(t,o,r);break;case 3:var s=t.updateQueue;if(s!==null){if(n=null,t.child!==null)switch(t.child.tag){case 5:n=t.child.stateNode;break;case 1:n=t.child.stateNode}Ku(t,s,n)}break;case 5:var a=t.stateNode;if(n===null&&t.flags&4){n=a;var l=t.memoizedProps;switch(t.type){case"button":case"input":case"select":case"textarea":l.autoFocus&&n.focus();break;case"img":l.src&&(n.src=l.src)}}break;case 6:break;case 4:break;case 12:break;case 13:if(t.memoizedState===null){var u=t.alternate;if(u!==null){var d=u.memoizedState;if(d!==null){var f=d.dehydrated;f!==null&&Lr(f)}}}break;case 19:case 17:case 21:case 22:case 23:case 25:break;default:throw Error(_(163))}ae||t.flags&512&&ka(t)}catch(m){K(t,t.return,m)}}if(t===e){C=null;break}if(n=t.sibling,n!==null){n.return=t.return,C=n;break}C=t.return}}function uc(e){for(;C!==null;){var t=C;if(t===e){C=null;break}var n=t.sibling;if(n!==null){n.return=t.return,C=n;break}C=t.return}}function cc(e){for(;C!==null;){var t=C;try{switch(t.tag){case 0:case 11:case 15:var n=t.return;try{Uo(4,t)}catch(l){K(t,n,l)}break;case 1:var r=t.stateNode;if(typeof r.componentDidMount=="function"){var i=t.return;try{r.componentDidMount()}catch(l){K(t,i,l)}}var o=t.return;try{ka(t)}catch(l){K(t,o,l)}break;case 5:var s=t.return;try{ka(t)}catch(l){K(t,s,l)}}}catch(l){K(t,t.return,l)}if(t===e){C=null;break}var a=t.sibling;if(a!==null){a.return=t.return,C=a;break}C=t.return}}var pv=Math.ceil,uo=mt.ReactCurrentDispatcher,Pl=mt.ReactCurrentOwner,Oe=mt.ReactCurrentBatchConfig,M=0,te=null,Q=null,re=0,_e=0,Ln=Ht(0),X=0,Hr=null,an=0,jo=0,Rl=0,kr=null,pe=null,Nl=0,Yn=1/0,Ze=null,co=!1,Pa=null,Mt=null,Ci=!1,Rt=null,fo=0,Cr=0,Ra=null,ji=-1,Fi=0;function de(){return M&6?q():ji!==-1?ji:ji=q()}function Ut(e){return e.mode&1?M&2&&re!==0?re&-re:Jg.transition!==null?(Fi===0&&(Fi=Kd()),Fi):(e=U,e!==0||(e=window.event,e=e===void 0?16:Zd(e.type)),e):1}function ze(e,t,n,r){if(50<Cr)throw Cr=0,Ra=null,Error(_(185));Jr(e,n,r),(!(M&2)||e!==te)&&(e===te&&(!(M&2)&&(jo|=n),X===4&&Et(e,re)),ye(e,r),n===1&&M===0&&!(t.mode&1)&&(Yn=q()+500,Lo&&Wt()))}function ye(e,t){var n=e.callbackNode;Jm(e,t);var r=qi(e,e===te?re:0);if(r===0)n!==null&&wu(n),e.callbackNode=null,e.callbackPriority=0;else if(t=r&-r,e.callbackPriority!==t){if(n!=null&&wu(n),t===1)e.tag===0?Yg(dc.bind(null,e)):wf(dc.bind(null,e)),Kg(function(){!(M&6)&&Wt()}),n=null;else{switch(Gd(r)){case 1:n=tl;break;case 4:n=Hd;break;case 16:n=Gi;break;case 536870912:n=Wd;break;default:n=Gi}n=vh(n,ch.bind(null,e))}e.callbackPriority=t,e.callbackNode=n}}function ch(e,t){if(ji=-1,Fi=0,M&6)throw Error(_(327));var n=e.callbackNode;if(bn()&&e.callbackNode!==n)return null;var r=qi(e,e===te?re:0);if(r===0)return null;if(r&30||r&e.expiredLanes||t)t=ho(e,r);else{t=r;var i=M;M|=2;var o=fh();(te!==e||re!==t)&&(Ze=null,Yn=q()+500,tn(e,t));do try{vv();break}catch(a){dh(e,a)}while(!0);pl(),uo.current=o,M=i,Q!==null?t=0:(te=null,re=0,t=X)}if(t!==0){if(t===2&&(i=ta(e),i!==0&&(r=i,t=Na(e,i))),t===1)throw n=Hr,tn(e,0),Et(e,r),ye(e,q()),n;if(t===6)Et(e,r);else{if(i=e.current.alternate,!(r&30)&&!mv(i)&&(t=ho(e,r),t===2&&(o=ta(e),o!==0&&(r=o,t=Na(e,o))),t===1))throw n=Hr,tn(e,0),Et(e,r),ye(e,q()),n;switch(e.finishedWork=i,e.finishedLanes=r,t){case 0:case 1:throw Error(_(345));case 2:Qt(e,pe,Ze);break;case 3:if(Et(e,r),(r&130023424)===r&&(t=Nl+500-q(),10<t)){if(qi(e,0)!==0)break;if(i=e.suspendedLanes,(i&r)!==r){de(),e.pingedLanes|=e.suspendedLanes&i;break}e.timeoutHandle=ua(Qt.bind(null,e,pe,Ze),t);break}Qt(e,pe,Ze);break;case 4:if(Et(e,r),(r&4194240)===r)break;for(t=e.eventTimes,i=-1;0<r;){var s=31-Fe(r);o=1<<s,s=t[s],s>i&&(i=s),r&=~o}if(r=i,r=q()-r,r=(120>r?120:480>r?480:1080>r?1080:1920>r?1920:3e3>r?3e3:4320>r?4320:1960*pv(r/1960))-r,10<r){e.timeoutHandle=ua(Qt.bind(null,e,pe,Ze),r);break}Qt(e,pe,Ze);break;case 5:Qt(e,pe,Ze);break;default:throw Error(_(329))}}}return ye(e,q()),e.callbackNode===n?ch.bind(null,e):null}function Na(e,t){var n=kr;return e.current.memoizedState.isDehydrated&&(tn(e,t).flags|=256),e=ho(e,t),e!==2&&(t=pe,pe=n,t!==null&&Oa(t)),e}function Oa(e){pe===null?pe=e:pe.push.apply(pe,e)}function mv(e){for(var t=e;;){if(t.flags&16384){var n=t.updateQueue;if(n!==null&&(n=n.stores,n!==null))for(var r=0;r<n.length;r++){var i=n[r],o=i.getSnapshot;i=i.value;try{if(!be(o(),i))return!1}catch{return!1}}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function Et(e,t){for(t&=~Rl,t&=~jo,e.suspendedLanes|=t,e.pingedLanes&=~t,e=e.expirationTimes;0<t;){var n=31-Fe(t),r=1<<n;e[n]=-1,t&=~r}}function dc(e){if(M&6)throw Error(_(327));bn();var t=qi(e,0);if(!(t&1))return ye(e,q()),null;var n=ho(e,t);if(e.tag!==0&&n===2){var r=ta(e);r!==0&&(t=r,n=Na(e,r))}if(n===1)throw n=Hr,tn(e,0),Et(e,t),ye(e,q()),n;if(n===6)throw Error(_(345));return e.finishedWork=e.current.alternate,e.finishedLanes=t,Qt(e,pe,Ze),ye(e,q()),null}function Ol(e,t){var n=M;M|=1;try{return e(t)}finally{M=n,M===0&&(Yn=q()+500,Lo&&Wt())}}function ln(e){Rt!==null&&Rt.tag===0&&!(M&6)&&bn();var t=M;M|=1;var n=Oe.transition,r=U;try{if(Oe.transition=null,U=1,e)return e()}finally{U=r,Oe.transition=n,M=t,!(M&6)&&Wt()}}function Al(){_e=Ln.current,$(Ln)}function tn(e,t){e.finishedWork=null,e.finishedLanes=0;var n=e.timeoutHandle;if(n!==-1&&(e.timeoutHandle=-1,Wg(n)),Q!==null)for(n=Q.return;n!==null;){var r=n;switch(dl(r),r.tag){case 1:r=r.type.childContextTypes,r!=null&&Zi();break;case 3:qn(),$(ge),$(le),_l();break;case 5:wl(r);break;case 4:qn();break;case 13:$(V);break;case 19:$(V);break;case 10:ml(r.type._context);break;case 22:case 23:Al()}n=n.return}if(te=e,Q=e=jt(e.current,null),re=_e=t,X=0,Hr=null,Rl=jo=an=0,pe=kr=null,Xt!==null){for(t=0;t<Xt.length;t++)if(n=Xt[t],r=n.interleaved,r!==null){n.interleaved=null;var i=r.next,o=n.pending;if(o!==null){var s=o.next;o.next=i,r.next=s}n.pending=r}Xt=null}return e}function dh(e,t){do{var n=Q;try{if(pl(),Di.current=lo,ao){for(var r=H.memoizedState;r!==null;){var i=r.queue;i!==null&&(i.pending=null),r=r.next}ao=!1}if(sn=0,ee=J=H=null,Er=!1,$r=0,Pl.current=null,n===null||n.return===null){X=1,Hr=t,Q=null;break}e:{var o=e,s=n.return,a=n,l=t;if(t=re,a.flags|=32768,l!==null&&typeof l=="object"&&typeof l.then=="function"){var u=l,d=a,f=d.tag;if(!(d.mode&1)&&(f===0||f===11||f===15)){var m=d.alternate;m?(d.updateQueue=m.updateQueue,d.memoizedState=m.memoizedState,d.lanes=m.lanes):(d.updateQueue=null,d.memoizedState=null)}var g=Xu(s);if(g!==null){g.flags&=-257,Zu(g,s,a,o,t),g.mode&1&&Ju(o,u,t),t=g,l=u;var y=t.updateQueue;if(y===null){var w=new Set;w.add(l),t.updateQueue=w}else y.add(l);break e}else{if(!(t&1)){Ju(o,u,t),xl();break e}l=Error(_(426))}}else if(B&&a.mode&1){var E=Xu(s);if(E!==null){!(E.flags&65536)&&(E.flags|=256),Zu(E,s,a,o,t),fl(Qn(l,a));break e}}o=l=Qn(l,a),X!==4&&(X=2),kr===null?kr=[o]:kr.push(o),o=s;do{switch(o.tag){case 3:o.flags|=65536,t&=-t,o.lanes|=t;var h=qf(o,l,t);Wu(o,h);break e;case 1:a=l;var c=o.type,p=o.stateNode;if(!(o.flags&128)&&(typeof c.getDerivedStateFromError=="function"||p!==null&&typeof p.componentDidCatch=="function"&&(Mt===null||!Mt.has(p)))){o.flags|=65536,t&=-t,o.lanes|=t;var v=Qf(o,a,t);Wu(o,v);break e}}o=o.return}while(o!==null)}ph(n)}catch(S){t=S,Q===n&&n!==null&&(Q=n=n.return);continue}break}while(!0)}function fh(){var e=uo.current;return uo.current=lo,e===null?lo:e}function xl(){(X===0||X===3||X===2)&&(X=4),te===null||!(an&268435455)&&!(jo&268435455)||Et(te,re)}function ho(e,t){var n=M;M|=2;var r=fh();(te!==e||re!==t)&&(Ze=null,tn(e,t));do try{gv();break}catch(i){dh(e,i)}while(!0);if(pl(),M=n,uo.current=r,Q!==null)throw Error(_(261));return te=null,re=0,X}function gv(){for(;Q!==null;)hh(Q)}function vv(){for(;Q!==null&&!Bm();)hh(Q)}function hh(e){var t=gh(e.alternate,e,_e);e.memoizedProps=e.pendingProps,t===null?ph(e):Q=t,Pl.current=null}function ph(e){var t=e;do{var n=t.alternate;if(e=t.return,t.flags&32768){if(n=cv(n,t),n!==null){n.flags&=32767,Q=n;return}if(e!==null)e.flags|=32768,e.subtreeFlags=0,e.deletions=null;else{X=6,Q=null;return}}else if(n=uv(n,t,_e),n!==null){Q=n;return}if(t=t.sibling,t!==null){Q=t;return}Q=t=e}while(t!==null);X===0&&(X=5)}function Qt(e,t,n){var r=U,i=Oe.transition;try{Oe.transition=null,U=1,yv(e,t,n,r)}finally{Oe.transition=i,U=r}return null}function yv(e,t,n,r){do bn();while(Rt!==null);if(M&6)throw Error(_(327));n=e.finishedWork;var i=e.finishedLanes;if(n===null)return null;if(e.finishedWork=null,e.finishedLanes=0,n===e.current)throw Error(_(177));e.callbackNode=null,e.callbackPriority=0;var o=n.lanes|n.childLanes;if(Xm(e,o),e===te&&(Q=te=null,re=0),!(n.subtreeFlags&2064)&&!(n.flags&2064)||Ci||(Ci=!0,vh(Gi,function(){return bn(),null})),o=(n.flags&15990)!==0,n.subtreeFlags&15990||o){o=Oe.transition,Oe.transition=null;var s=U;U=1;var a=M;M|=4,Pl.current=null,fv(e,n),lh(n,e),Fg(aa),Qi=!!sa,aa=sa=null,e.current=n,hv(n),Vm(),M=a,U=s,Oe.transition=o}else e.current=n;if(Ci&&(Ci=!1,Rt=e,fo=i),o=e.pendingLanes,o===0&&(Mt=null),Km(n.stateNode),ye(e,q()),t!==null)for(r=e.onRecoverableError,n=0;n<t.length;n++)i=t[n],r(i.value,{componentStack:i.stack,digest:i.digest});if(co)throw co=!1,e=Pa,Pa=null,e;return fo&1&&e.tag!==0&&bn(),o=e.pendingLanes,o&1?e===Ra?Cr++:(Cr=0,Ra=e):Cr=0,Wt(),null}function bn(){if(Rt!==null){var e=Gd(fo),t=Oe.transition,n=U;try{if(Oe.transition=null,U=16>e?16:e,Rt===null)var r=!1;else{if(e=Rt,Rt=null,fo=0,M&6)throw Error(_(331));var i=M;for(M|=4,C=e.current;C!==null;){var o=C,s=o.child;if(C.flags&16){var a=o.deletions;if(a!==null){for(var l=0;l<a.length;l++){var u=a[l];for(C=u;C!==null;){var d=C;switch(d.tag){case 0:case 11:case 15:Ir(8,d,o)}var f=d.child;if(f!==null)f.return=d,C=f;else for(;C!==null;){d=C;var m=d.sibling,g=d.return;if(oh(d),d===u){C=null;break}if(m!==null){m.return=g,C=m;break}C=g}}}var y=o.alternate;if(y!==null){var w=y.child;if(w!==null){y.child=null;do{var E=w.sibling;w.sibling=null,w=E}while(w!==null)}}C=o}}if(o.subtreeFlags&2064&&s!==null)s.return=o,C=s;else e:for(;C!==null;){if(o=C,o.flags&2048)switch(o.tag){case 0:case 11:case 15:Ir(9,o,o.return)}var h=o.sibling;if(h!==null){h.return=o.return,C=h;break e}C=o.return}}var c=e.current;for(C=c;C!==null;){s=C;var p=s.child;if(s.subtreeFlags&2064&&p!==null)p.return=s,C=p;else e:for(s=c;C!==null;){if(a=C,a.flags&2048)try{switch(a.tag){case 0:case 11:case 15:Uo(9,a)}}catch(S){K(a,a.return,S)}if(a===s){C=null;break e}var v=a.sibling;if(v!==null){v.return=a.return,C=v;break e}C=a.return}}if(M=i,Wt(),We&&typeof We.onPostCommitFiberRoot=="function")try{We.onPostCommitFiberRoot(Ro,e)}catch{}r=!0}return r}finally{U=n,Oe.transition=t}}return!1}function fc(e,t,n){t=Qn(n,t),t=qf(e,t,1),e=Dt(e,t,1),t=de(),e!==null&&(Jr(e,1,t),ye(e,t))}function K(e,t,n){if(e.tag===3)fc(e,e,n);else for(;t!==null;){if(t.tag===3){fc(t,e,n);break}else if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError=="function"||typeof r.componentDidCatch=="function"&&(Mt===null||!Mt.has(r))){e=Qn(n,e),e=Qf(t,e,1),t=Dt(t,e,1),e=de(),t!==null&&(Jr(t,1,e),ye(t,e));break}}t=t.return}}function wv(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),t=de(),e.pingedLanes|=e.suspendedLanes&n,te===e&&(re&n)===n&&(X===4||X===3&&(re&130023424)===re&&500>q()-Nl?tn(e,0):Rl|=n),ye(e,t)}function mh(e,t){t===0&&(e.mode&1?(t=mi,mi<<=1,!(mi&130023424)&&(mi=4194304)):t=1);var n=de();e=dt(e,t),e!==null&&(Jr(e,t,n),ye(e,n))}function _v(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),mh(e,n)}function Sv(e,t){var n=0;switch(e.tag){case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;default:throw Error(_(314))}r!==null&&r.delete(t),mh(e,n)}var gh;gh=function(e,t,n){if(e!==null)if(e.memoizedProps!==t.pendingProps||ge.current)me=!0;else{if(!(e.lanes&n)&&!(t.flags&128))return me=!1,lv(e,t,n);me=!!(e.flags&131072)}else me=!1,B&&t.flags&1048576&&_f(t,no,t.index);switch(t.lanes=0,t.tag){case 2:var r=t.type;Ui(e,t),e=t.pendingProps;var i=Wn(t,le.current);zn(t,n),i=El(null,t,r,e,i,n);var o=Il();return t.flags|=1,typeof i=="object"&&i!==null&&typeof i.render=="function"&&i.$$typeof===void 0?(t.tag=1,t.memoizedState=null,t.updateQueue=null,ve(r)?(o=!0,eo(t)):o=!1,t.memoizedState=i.state!==null&&i.state!==void 0?i.state:null,vl(t),i.updater=Mo,t.stateNode=i,i._reactInternals=t,ga(t,r,e,n),t=wa(null,t,r,!0,o,n)):(t.tag=0,B&&o&&cl(t),ce(null,t,i,n),t=t.child),t;case 16:r=t.elementType;e:{switch(Ui(e,t),e=t.pendingProps,i=r._init,r=i(r._payload),t.type=r,i=t.tag=Iv(r),e=Me(r,e),i){case 0:t=ya(null,t,r,e,n);break e;case 1:t=nc(null,t,r,e,n);break e;case 11:t=ec(null,t,r,e,n);break e;case 14:t=tc(null,t,r,Me(r.type,e),n);break e}throw Error(_(306,r,""))}return t;case 0:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Me(r,i),ya(e,t,r,i,n);case 1:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Me(r,i),nc(e,t,r,i,n);case 3:e:{if(Zf(t),e===null)throw Error(_(387));r=t.pendingProps,o=t.memoizedState,i=o.element,Tf(e,t),oo(t,r,null,n);var s=t.memoizedState;if(r=s.element,o.isDehydrated)if(o={element:r,isDehydrated:!1,cache:s.cache,pendingSuspenseBoundaries:s.pendingSuspenseBoundaries,transitions:s.transitions},t.updateQueue.baseState=o,t.memoizedState=o,t.flags&256){i=Qn(Error(_(423)),t),t=rc(e,t,r,n,i);break e}else if(r!==i){i=Qn(Error(_(424)),t),t=rc(e,t,r,n,i);break e}else for(Se=Lt(t.stateNode.containerInfo.firstChild),Ee=t,B=!0,je=null,n=kf(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|4096,n=n.sibling;else{if(Kn(),r===i){t=ft(e,t,n);break e}ce(e,t,r,n)}t=t.child}return t;case 5:return Pf(t),e===null&&ha(t),r=t.type,i=t.pendingProps,o=e!==null?e.memoizedProps:null,s=i.children,la(r,i)?s=null:o!==null&&la(r,o)&&(t.flags|=32),Xf(e,t),ce(e,t,s,n),t.child;case 6:return e===null&&ha(t),null;case 13:return eh(e,t,n);case 4:return yl(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=Gn(t,null,r,n):ce(e,t,r,n),t.child;case 11:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Me(r,i),ec(e,t,r,i,n);case 7:return ce(e,t,t.pendingProps,n),t.child;case 8:return ce(e,t,t.pendingProps.children,n),t.child;case 12:return ce(e,t,t.pendingProps.children,n),t.child;case 10:e:{if(r=t.type._context,i=t.pendingProps,o=t.memoizedProps,s=i.value,F(ro,r._currentValue),r._currentValue=s,o!==null)if(be(o.value,s)){if(o.children===i.children&&!ge.current){t=ft(e,t,n);break e}}else for(o=t.child,o!==null&&(o.return=t);o!==null;){var a=o.dependencies;if(a!==null){s=o.child;for(var l=a.firstContext;l!==null;){if(l.context===r){if(o.tag===1){l=at(-1,n&-n),l.tag=2;var u=o.updateQueue;if(u!==null){u=u.shared;var d=u.pending;d===null?l.next=l:(l.next=d.next,d.next=l),u.pending=l}}o.lanes|=n,l=o.alternate,l!==null&&(l.lanes|=n),pa(o.return,n,t),a.lanes|=n;break}l=l.next}}else if(o.tag===10)s=o.type===t.type?null:o.child;else if(o.tag===18){if(s=o.return,s===null)throw Error(_(341));s.lanes|=n,a=s.alternate,a!==null&&(a.lanes|=n),pa(s,n,t),s=o.sibling}else s=o.child;if(s!==null)s.return=o;else for(s=o;s!==null;){if(s===t){s=null;break}if(o=s.sibling,o!==null){o.return=s.return,s=o;break}s=s.return}o=s}ce(e,t,i.children,n),t=t.child}return t;case 9:return i=t.type,r=t.pendingProps.children,zn(t,n),i=Ae(i),r=r(i),t.flags|=1,ce(e,t,r,n),t.child;case 14:return r=t.type,i=Me(r,t.pendingProps),i=Me(r.type,i),tc(e,t,r,i,n);case 15:return Yf(e,t,t.type,t.pendingProps,n);case 17:return r=t.type,i=t.pendingProps,i=t.elementType===r?i:Me(r,i),Ui(e,t),t.tag=1,ve(r)?(e=!0,eo(t)):e=!1,zn(t,n),Gf(t,r,i),ga(t,r,i,n),wa(null,t,r,!0,e,n);case 19:return th(e,t,n);case 22:return Jf(e,t,n)}throw Error(_(156,t.tag))};function vh(e,t){return Vd(e,t)}function Ev(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Ne(e,t,n,r){return new Ev(e,t,n,r)}function Ll(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Iv(e){if(typeof e=="function")return Ll(e)?1:0;if(e!=null){if(e=e.$$typeof,e===Xa)return 11;if(e===Za)return 14}return 2}function jt(e,t){var n=e.alternate;return n===null?(n=Ne(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&14680064,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}function zi(e,t,n,r,i,o){var s=2;if(r=e,typeof e=="function")Ll(e)&&(s=1);else if(typeof e=="string")s=5;else e:switch(e){case In:return nn(n.children,i,o,t);case Ja:s=8,i|=8;break;case bs:return e=Ne(12,n,t,i|2),e.elementType=bs,e.lanes=o,e;case $s:return e=Ne(13,n,t,i),e.elementType=$s,e.lanes=o,e;case Bs:return e=Ne(19,n,t,i),e.elementType=Bs,e.lanes=o,e;case Td:return Fo(n,i,o,t);default:if(typeof e=="object"&&e!==null)switch(e.$$typeof){case kd:s=10;break e;case Cd:s=9;break e;case Xa:s=11;break e;case Za:s=14;break e;case wt:s=16,r=null;break e}throw Error(_(130,e==null?e:typeof e,""))}return t=Ne(s,n,t,i),t.elementType=e,t.type=r,t.lanes=o,t}function nn(e,t,n,r){return e=Ne(7,e,r,t),e.lanes=n,e}function Fo(e,t,n,r){return e=Ne(22,e,r,t),e.elementType=Td,e.lanes=n,e.stateNode={isHidden:!1},e}function Cs(e,t,n){return e=Ne(6,e,null,t),e.lanes=n,e}function Ts(e,t,n){return t=Ne(4,e.children!==null?e.children:[],e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}function kv(e,t,n,r,i){this.tag=t,this.containerInfo=e,this.finishedWork=this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.pendingContext=this.context=null,this.callbackPriority=0,this.eventTimes=ss(0),this.expirationTimes=ss(-1),this.entangledLanes=this.finishedLanes=this.mutableReadLanes=this.expiredLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=ss(0),this.identifierPrefix=r,this.onRecoverableError=i,this.mutableSourceEagerHydrationData=null}function Dl(e,t,n,r,i,o,s,a,l){return e=new kv(e,t,n,a,l),t===1?(t=1,o===!0&&(t|=8)):t=0,o=Ne(3,null,null,t),e.current=o,o.stateNode=e,o.memoizedState={element:r,isDehydrated:n,cache:null,transitions:null,pendingSuspenseBoundaries:null},vl(o),e}function Cv(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:En,key:r==null?null:""+r,children:e,containerInfo:t,implementation:n}}function yh(e){if(!e)return Bt;e=e._reactInternals;e:{if(mn(e)!==e||e.tag!==1)throw Error(_(170));var t=e;do{switch(t.tag){case 3:t=t.stateNode.context;break e;case 1:if(ve(t.type)){t=t.stateNode.__reactInternalMemoizedMergedChildContext;break e}}t=t.return}while(t!==null);throw Error(_(171))}if(e.tag===1){var n=e.type;if(ve(n))return yf(e,n,t)}return t}function wh(e,t,n,r,i,o,s,a,l){return e=Dl(n,r,!0,e,i,o,s,a,l),e.context=yh(null),n=e.current,r=de(),i=Ut(n),o=at(r,i),o.callback=t??null,Dt(n,o,i),e.current.lanes=i,Jr(e,i,r),ye(e,r),e}function zo(e,t,n,r){var i=t.current,o=de(),s=Ut(i);return n=yh(n),t.context===null?t.context=n:t.pendingContext=n,t=at(o,s),t.payload={element:e},r=r===void 0?null:r,r!==null&&(t.callback=r),e=Dt(i,t,s),e!==null&&(ze(e,i,s,o),Li(e,i,s)),s}function po(e){if(e=e.current,!e.child)return null;switch(e.child.tag){case 5:return e.child.stateNode;default:return e.child.stateNode}}function hc(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function Ml(e,t){hc(e,t),(e=e.alternate)&&hc(e,t)}function Tv(){return null}var _h=typeof reportError=="function"?reportError:function(e){console.error(e)};function Ul(e){this._internalRoot=e}bo.prototype.render=Ul.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(_(409));zo(e,t,null,null)};bo.prototype.unmount=Ul.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;ln(function(){zo(null,e,null,null)}),t[ct]=null}};function bo(e){this._internalRoot=e}bo.prototype.unstable_scheduleHydration=function(e){if(e){var t=Yd();e={blockedOn:null,target:e,priority:t};for(var n=0;n<St.length&&t!==0&&t<St[n].priority;n++);St.splice(n,0,e),n===0&&Xd(e)}};function jl(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function $o(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11&&(e.nodeType!==8||e.nodeValue!==" react-mount-point-unstable "))}function pc(){}function Pv(e,t,n,r,i){if(i){if(typeof r=="function"){var o=r;r=function(){var u=po(s);o.call(u)}}var s=wh(t,r,e,0,null,!1,!1,"",pc);return e._reactRootContainer=s,e[ct]=s.current,Ur(e.nodeType===8?e.parentNode:e),ln(),s}for(;i=e.lastChild;)e.removeChild(i);if(typeof r=="function"){var a=r;r=function(){var u=po(l);a.call(u)}}var l=Dl(e,0,!1,null,null,!1,!1,"",pc);return e._reactRootContainer=l,e[ct]=l.current,Ur(e.nodeType===8?e.parentNode:e),ln(function(){zo(t,l,n,r)}),l}function Bo(e,t,n,r,i){var o=n._reactRootContainer;if(o){var s=o;if(typeof i=="function"){var a=i;i=function(){var l=po(s);a.call(l)}}zo(t,s,e,i)}else s=Pv(n,t,e,i,r);return po(s)}qd=function(e){switch(e.tag){case 3:var t=e.stateNode;if(t.current.memoizedState.isDehydrated){var n=pr(t.pendingLanes);n!==0&&(nl(t,n|1),ye(t,q()),!(M&6)&&(Yn=q()+500,Wt()))}break;case 13:ln(function(){var r=dt(e,1);if(r!==null){var i=de();ze(r,e,1,i)}}),Ml(e,1)}};rl=function(e){if(e.tag===13){var t=dt(e,134217728);if(t!==null){var n=de();ze(t,e,134217728,n)}Ml(e,134217728)}};Qd=function(e){if(e.tag===13){var t=Ut(e),n=dt(e,t);if(n!==null){var r=de();ze(n,e,t,r)}Ml(e,t)}};Yd=function(){return U};Jd=function(e,t){var n=U;try{return U=e,t()}finally{U=n}};Xs=function(e,t,n){switch(t){case"input":if(Ws(e,n),t=n.name,n.type==="radio"&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll("input[name="+JSON.stringify(""+t)+'][type="radio"]'),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=xo(r);if(!i)throw Error(_(90));Rd(r),Ws(r,i)}}}break;case"textarea":Od(e,n);break;case"select":t=n.value,t!=null&&Mn(e,!!n.multiple,t,!1)}};jd=Ol;Fd=ln;var Rv={usingClientEntryPoint:!1,Events:[Zr,Pn,xo,Md,Ud,Ol]},dr={findFiberByHostInstance:Jt,bundleType:0,version:"18.3.1",rendererPackageName:"react-dom"},Nv={bundleType:dr.bundleType,version:dr.version,rendererPackageName:dr.rendererPackageName,rendererConfig:dr.rendererConfig,overrideHookState:null,overrideHookStateDeletePath:null,overrideHookStateRenamePath:null,overrideProps:null,overridePropsDeletePath:null,overridePropsRenamePath:null,setErrorHandler:null,setSuspenseHandler:null,scheduleUpdate:null,currentDispatcherRef:mt.ReactCurrentDispatcher,findHostInstanceByFiber:function(e){return e=$d(e),e===null?null:e.stateNode},findFiberByHostInstance:dr.findFiberByHostInstance||Tv,findHostInstancesForRefresh:null,scheduleRefresh:null,scheduleRoot:null,setRefreshHandler:null,getCurrentFiber:null,reconcilerVersion:"18.3.1-next-f1338f8080-20240426"};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<"u"){var Ti=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Ti.isDisabled&&Ti.supportsFiber)try{Ro=Ti.inject(Nv),We=Ti}catch{}}Ce.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED=Rv;Ce.createPortal=function(e,t){var n=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!jl(t))throw Error(_(200));return Cv(e,t,null,n)};Ce.createRoot=function(e,t){if(!jl(e))throw Error(_(299));var n=!1,r="",i=_h;return t!=null&&(t.unstable_strictMode===!0&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onRecoverableError!==void 0&&(i=t.onRecoverableError)),t=Dl(e,1,!1,null,null,n,!1,r,i),e[ct]=t.current,Ur(e.nodeType===8?e.parentNode:e),new Ul(t)};Ce.findDOMNode=function(e){if(e==null)return null;if(e.nodeType===1)return e;var t=e._reactInternals;if(t===void 0)throw typeof e.render=="function"?Error(_(188)):(e=Object.keys(e).join(","),Error(_(268,e)));return e=$d(t),e=e===null?null:e.stateNode,e};Ce.flushSync=function(e){return ln(e)};Ce.hydrate=function(e,t,n){if(!$o(t))throw Error(_(200));return Bo(null,e,t,!0,n)};Ce.hydrateRoot=function(e,t,n){if(!jl(e))throw Error(_(405));var r=n!=null&&n.hydratedSources||null,i=!1,o="",s=_h;if(n!=null&&(n.unstable_strictMode===!0&&(i=!0),n.identifierPrefix!==void 0&&(o=n.identifierPrefix),n.onRecoverableError!==void 0&&(s=n.onRecoverableError)),t=wh(t,null,e,1,n??null,i,!1,o,s),e[ct]=t.current,Ur(e),r)for(e=0;e<r.length;e++)n=r[e],i=n._getVersion,i=i(n._source),t.mutableSourceEagerHydrationData==null?t.mutableSourceEagerHydrationData=[n,i]:t.mutableSourceEagerHydrationData.push(n,i);return new bo(t)};Ce.render=function(e,t,n){if(!$o(t))throw Error(_(200));return Bo(null,e,t,!1,n)};Ce.unmountComponentAtNode=function(e){if(!$o(e))throw Error(_(40));return e._reactRootContainer?(ln(function(){Bo(null,null,e,!1,function(){e._reactRootContainer=null,e[ct]=null})}),!0):!1};Ce.unstable_batchedUpdates=Ol;Ce.unstable_renderSubtreeIntoContainer=function(e,t,n,r){if(!$o(n))throw Error(_(200));if(e==null||e._reactInternals===void 0)throw Error(_(38));return Bo(e,t,n,!1,r)};Ce.version="18.3.1-next-f1338f8080-20240426";function Sh(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>"u"||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!="function"))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Sh)}catch(e){console.error(e)}}Sh(),_d.exports=Ce;var Ov=_d.exports,mc=Ov;Fs.createRoot=mc.createRoot,Fs.hydrateRoot=mc.hydrateRoot;const Fl="/assets/fns_extract-zIBugqUX.png",Eh="/assets/FNS-ChHkiTOW.png";/**
 * @remix-run/router v1.16.1
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Wr(){return Wr=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Wr.apply(this,arguments)}var Nt;(function(e){e.Pop="POP",e.Push="PUSH",e.Replace="REPLACE"})(Nt||(Nt={}));const gc="popstate";function Av(e){e===void 0&&(e={});function t(r,i){let{pathname:o,search:s,hash:a}=r.location;return Aa("",{pathname:o,search:s,hash:a},i.state&&i.state.usr||null,i.state&&i.state.key||"default")}function n(r,i){return typeof i=="string"?i:mo(i)}return Lv(t,n,null,e)}function Y(e,t){if(e===!1||e===null||typeof e>"u")throw new Error(t)}function Ih(e,t){if(!e){typeof console<"u"&&console.warn(t);try{throw new Error(t)}catch{}}}function xv(){return Math.random().toString(36).substr(2,8)}function vc(e,t){return{usr:e.state,key:e.key,idx:t}}function Aa(e,t,n,r){return n===void 0&&(n=null),Wr({pathname:typeof e=="string"?e:e.pathname,search:"",hash:""},typeof t=="string"?er(t):t,{state:n,key:t&&t.key||r||xv()})}function mo(e){let{pathname:t="/",search:n="",hash:r=""}=e;return n&&n!=="?"&&(t+=n.charAt(0)==="?"?n:"?"+n),r&&r!=="#"&&(t+=r.charAt(0)==="#"?r:"#"+r),t}function er(e){let t={};if(e){let n=e.indexOf("#");n>=0&&(t.hash=e.substr(n),e=e.substr(0,n));let r=e.indexOf("?");r>=0&&(t.search=e.substr(r),e=e.substr(0,r)),e&&(t.pathname=e)}return t}function Lv(e,t,n,r){r===void 0&&(r={});let{window:i=document.defaultView,v5Compat:o=!1}=r,s=i.history,a=Nt.Pop,l=null,u=d();u==null&&(u=0,s.replaceState(Wr({},s.state,{idx:u}),""));function d(){return(s.state||{idx:null}).idx}function f(){a=Nt.Pop;let E=d(),h=E==null?null:E-u;u=E,l&&l({action:a,location:w.location,delta:h})}function m(E,h){a=Nt.Push;let c=Aa(w.location,E,h);u=d()+1;let p=vc(c,u),v=w.createHref(c);try{s.pushState(p,"",v)}catch(S){if(S instanceof DOMException&&S.name==="DataCloneError")throw S;i.location.assign(v)}o&&l&&l({action:a,location:w.location,delta:1})}function g(E,h){a=Nt.Replace;let c=Aa(w.location,E,h);u=d();let p=vc(c,u),v=w.createHref(c);s.replaceState(p,"",v),o&&l&&l({action:a,location:w.location,delta:0})}function y(E){let h=i.location.origin!=="null"?i.location.origin:i.location.href,c=typeof E=="string"?E:mo(E);return c=c.replace(/ $/,"%20"),Y(h,"No window.location.(origin|href) available to create URL for href: "+c),new URL(c,h)}let w={get action(){return a},get location(){return e(i,s)},listen(E){if(l)throw new Error("A history only accepts one active listener");return i.addEventListener(gc,f),l=E,()=>{i.removeEventListener(gc,f),l=null}},createHref(E){return t(i,E)},createURL:y,encodeLocation(E){let h=y(E);return{pathname:h.pathname,search:h.search,hash:h.hash}},push:m,replace:g,go(E){return s.go(E)}};return w}var yc;(function(e){e.data="data",e.deferred="deferred",e.redirect="redirect",e.error="error"})(yc||(yc={}));function Dv(e,t,n){n===void 0&&(n="/");let r=typeof t=="string"?er(t):t,i=zl(r.pathname||"/",n);if(i==null)return null;let o=kh(e);Mv(o);let s=null;for(let a=0;s==null&&a<o.length;++a){let l=Gv(i);s=Hv(o[a],l)}return s}function kh(e,t,n,r){t===void 0&&(t=[]),n===void 0&&(n=[]),r===void 0&&(r="");let i=(o,s,a)=>{let l={relativePath:a===void 0?o.path||"":a,caseSensitive:o.caseSensitive===!0,childrenIndex:s,route:o};l.relativePath.startsWith("/")&&(Y(l.relativePath.startsWith(r),'Absolute route path "'+l.relativePath+'" nested under path '+('"'+r+'" is not valid. An absolute child route path ')+"must start with the combined path of all its parent routes."),l.relativePath=l.relativePath.slice(r.length));let u=Ft([r,l.relativePath]),d=n.concat(l);o.children&&o.children.length>0&&(Y(o.index!==!0,"Index routes must not have child routes. Please remove "+('all child routes from route path "'+u+'".')),kh(o.children,t,d,u)),!(o.path==null&&!o.index)&&t.push({path:u,score:Bv(u,o.index),routesMeta:d})};return e.forEach((o,s)=>{var a;if(o.path===""||!((a=o.path)!=null&&a.includes("?")))i(o,s);else for(let l of Ch(o.path))i(o,s,l)}),t}function Ch(e){let t=e.split("/");if(t.length===0)return[];let[n,...r]=t,i=n.endsWith("?"),o=n.replace(/\?$/,"");if(r.length===0)return i?[o,""]:[o];let s=Ch(r.join("/")),a=[];return a.push(...s.map(l=>l===""?o:[o,l].join("/"))),i&&a.push(...s),a.map(l=>e.startsWith("/")&&l===""?"/":l)}function Mv(e){e.sort((t,n)=>t.score!==n.score?n.score-t.score:Vv(t.routesMeta.map(r=>r.childrenIndex),n.routesMeta.map(r=>r.childrenIndex)))}const Uv=/^:[\w-]+$/,jv=3,Fv=2,zv=1,bv=10,$v=-2,wc=e=>e==="*";function Bv(e,t){let n=e.split("/"),r=n.length;return n.some(wc)&&(r+=$v),t&&(r+=Fv),n.filter(i=>!wc(i)).reduce((i,o)=>i+(Uv.test(o)?jv:o===""?zv:bv),r)}function Vv(e,t){return e.length===t.length&&e.slice(0,-1).every((r,i)=>r===t[i])?e[e.length-1]-t[t.length-1]:0}function Hv(e,t){let{routesMeta:n}=e,r={},i="/",o=[];for(let s=0;s<n.length;++s){let a=n[s],l=s===n.length-1,u=i==="/"?t:t.slice(i.length)||"/",d=Wv({path:a.relativePath,caseSensitive:a.caseSensitive,end:l},u);if(!d)return null;Object.assign(r,d.params);let f=a.route;o.push({params:r,pathname:Ft([i,d.pathname]),pathnameBase:Jv(Ft([i,d.pathnameBase])),route:f}),d.pathnameBase!=="/"&&(i=Ft([i,d.pathnameBase]))}return o}function Wv(e,t){typeof e=="string"&&(e={path:e,caseSensitive:!1,end:!0});let[n,r]=Kv(e.path,e.caseSensitive,e.end),i=t.match(n);if(!i)return null;let o=i[0],s=o.replace(/(.)\/+$/,"$1"),a=i.slice(1);return{params:r.reduce((u,d,f)=>{let{paramName:m,isOptional:g}=d;if(m==="*"){let w=a[f]||"";s=o.slice(0,o.length-w.length).replace(/(.)\/+$/,"$1")}const y=a[f];return g&&!y?u[m]=void 0:u[m]=(y||"").replace(/%2F/g,"/"),u},{}),pathname:o,pathnameBase:s,pattern:e}}function Kv(e,t,n){t===void 0&&(t=!1),n===void 0&&(n=!0),Ih(e==="*"||!e.endsWith("*")||e.endsWith("/*"),'Route path "'+e+'" will be treated as if it were '+('"'+e.replace(/\*$/,"/*")+'" because the `*` character must ')+"always follow a `/` in the pattern. To get rid of this warning, "+('please change the route path to "'+e.replace(/\*$/,"/*")+'".'));let r=[],i="^"+e.replace(/\/*\*?$/,"").replace(/^\/*/,"/").replace(/[\\.*+^${}|()[\]]/g,"\\$&").replace(/\/:([\w-]+)(\?)?/g,(s,a,l)=>(r.push({paramName:a,isOptional:l!=null}),l?"/?([^\\/]+)?":"/([^\\/]+)"));return e.endsWith("*")?(r.push({paramName:"*"}),i+=e==="*"||e==="/*"?"(.*)$":"(?:\\/(.+)|\\/*)$"):n?i+="\\/*$":e!==""&&e!=="/"&&(i+="(?:(?=\\/|$))"),[new RegExp(i,t?void 0:"i"),r]}function Gv(e){try{return e.split("/").map(t=>decodeURIComponent(t).replace(/\//g,"%2F")).join("/")}catch(t){return Ih(!1,'The URL path "'+e+'" could not be decoded because it is is a malformed URL segment. This is probably due to a bad percent '+("encoding ("+t+").")),e}}function zl(e,t){if(t==="/")return e;if(!e.toLowerCase().startsWith(t.toLowerCase()))return null;let n=t.endsWith("/")?t.length-1:t.length,r=e.charAt(n);return r&&r!=="/"?null:e.slice(n)||"/"}function qv(e,t){t===void 0&&(t="/");let{pathname:n,search:r="",hash:i=""}=typeof e=="string"?er(e):e;return{pathname:n?n.startsWith("/")?n:Qv(n,t):t,search:Xv(r),hash:Zv(i)}}function Qv(e,t){let n=t.replace(/\/+$/,"").split("/");return e.split("/").forEach(i=>{i===".."?n.length>1&&n.pop():i!=="."&&n.push(i)}),n.length>1?n.join("/"):"/"}function Ps(e,t,n,r){return"Cannot include a '"+e+"' character in a manually specified "+("`to."+t+"` field ["+JSON.stringify(r)+"].  Please separate it out to the ")+("`to."+n+"` field. Alternatively you may provide the full path as ")+'a string in <Link to="..."> and the router will parse it for you.'}function Yv(e){return e.filter((t,n)=>n===0||t.route.path&&t.route.path.length>0)}function Th(e,t){let n=Yv(e);return t?n.map((r,i)=>i===e.length-1?r.pathname:r.pathnameBase):n.map(r=>r.pathnameBase)}function Ph(e,t,n,r){r===void 0&&(r=!1);let i;typeof e=="string"?i=er(e):(i=Wr({},e),Y(!i.pathname||!i.pathname.includes("?"),Ps("?","pathname","search",i)),Y(!i.pathname||!i.pathname.includes("#"),Ps("#","pathname","hash",i)),Y(!i.search||!i.search.includes("#"),Ps("#","search","hash",i)));let o=e===""||i.pathname==="",s=o?"/":i.pathname,a;if(s==null)a=n;else{let f=t.length-1;if(!r&&s.startsWith("..")){let m=s.split("/");for(;m[0]==="..";)m.shift(),f-=1;i.pathname=m.join("/")}a=f>=0?t[f]:"/"}let l=qv(i,a),u=s&&s!=="/"&&s.endsWith("/"),d=(o||s===".")&&n.endsWith("/");return!l.pathname.endsWith("/")&&(u||d)&&(l.pathname+="/"),l}const Ft=e=>e.join("/").replace(/\/\/+/g,"/"),Jv=e=>e.replace(/\/+$/,"").replace(/^\/*/,"/"),Xv=e=>!e||e==="?"?"":e.startsWith("?")?e:"?"+e,Zv=e=>!e||e==="#"?"":e.startsWith("#")?e:"#"+e;function e0(e){return e!=null&&typeof e.status=="number"&&typeof e.statusText=="string"&&typeof e.internal=="boolean"&&"data"in e}const Rh=["post","put","patch","delete"];new Set(Rh);const t0=["get",...Rh];new Set(t0);/**
 * React Router v6.23.1
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function Kr(){return Kr=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Kr.apply(this,arguments)}const bl=I.createContext(null),n0=I.createContext(null),gn=I.createContext(null),Vo=I.createContext(null),vn=I.createContext({outlet:null,matches:[],isDataRoute:!1}),Nh=I.createContext(null);function r0(e,t){let{relative:n}=t===void 0?{}:t;ti()||Y(!1);let{basename:r,navigator:i}=I.useContext(gn),{hash:o,pathname:s,search:a}=Ah(e,{relative:n}),l=s;return r!=="/"&&(l=s==="/"?r:Ft([r,s])),i.createHref({pathname:l,search:a,hash:o})}function ti(){return I.useContext(Vo)!=null}function Ho(){return ti()||Y(!1),I.useContext(Vo).location}function Oh(e){I.useContext(gn).static||I.useLayoutEffect(e)}function $l(){let{isDataRoute:e}=I.useContext(vn);return e?g0():i0()}function i0(){ti()||Y(!1);let e=I.useContext(bl),{basename:t,future:n,navigator:r}=I.useContext(gn),{matches:i}=I.useContext(vn),{pathname:o}=Ho(),s=JSON.stringify(Th(i,n.v7_relativeSplatPath)),a=I.useRef(!1);return Oh(()=>{a.current=!0}),I.useCallback(function(u,d){if(d===void 0&&(d={}),!a.current)return;if(typeof u=="number"){r.go(u);return}let f=Ph(u,JSON.parse(s),o,d.relative==="path");e==null&&t!=="/"&&(f.pathname=f.pathname==="/"?t:Ft([t,f.pathname])),(d.replace?r.replace:r.push)(f,d.state,d)},[t,r,s,o,e])}function Ah(e,t){let{relative:n}=t===void 0?{}:t,{future:r}=I.useContext(gn),{matches:i}=I.useContext(vn),{pathname:o}=Ho(),s=JSON.stringify(Th(i,r.v7_relativeSplatPath));return I.useMemo(()=>Ph(e,JSON.parse(s),o,n==="path"),[e,s,o,n])}function o0(e,t){return s0(e,t)}function s0(e,t,n,r){ti()||Y(!1);let{navigator:i}=I.useContext(gn),{matches:o}=I.useContext(vn),s=o[o.length-1],a=s?s.params:{};s&&s.pathname;let l=s?s.pathnameBase:"/";s&&s.route;let u=Ho(),d;if(t){var f;let E=typeof t=="string"?er(t):t;l==="/"||(f=E.pathname)!=null&&f.startsWith(l)||Y(!1),d=E}else d=u;let m=d.pathname||"/",g=m;if(l!=="/"){let E=l.replace(/^\//,"").split("/");g="/"+m.replace(/^\//,"").split("/").slice(E.length).join("/")}let y=Dv(e,{pathname:g}),w=d0(y&&y.map(E=>Object.assign({},E,{params:Object.assign({},a,E.params),pathname:Ft([l,i.encodeLocation?i.encodeLocation(E.pathname).pathname:E.pathname]),pathnameBase:E.pathnameBase==="/"?l:Ft([l,i.encodeLocation?i.encodeLocation(E.pathnameBase).pathname:E.pathnameBase])})),o,n,r);return t&&w?I.createElement(Vo.Provider,{value:{location:Kr({pathname:"/",search:"",hash:"",state:null,key:"default"},d),navigationType:Nt.Pop}},w):w}function a0(){let e=m0(),t=e0(e)?e.status+" "+e.statusText:e instanceof Error?e.message:JSON.stringify(e),n=e instanceof Error?e.stack:null,i={padding:"0.5rem",backgroundColor:"rgba(200,200,200, 0.5)"};return I.createElement(I.Fragment,null,I.createElement("h2",null,"Unexpected Application Error!"),I.createElement("h3",{style:{fontStyle:"italic"}},t),n?I.createElement("pre",{style:i},n):null,null)}const l0=I.createElement(a0,null);class u0 extends I.Component{constructor(t){super(t),this.state={location:t.location,revalidation:t.revalidation,error:t.error}}static getDerivedStateFromError(t){return{error:t}}static getDerivedStateFromProps(t,n){return n.location!==t.location||n.revalidation!=="idle"&&t.revalidation==="idle"?{error:t.error,location:t.location,revalidation:t.revalidation}:{error:t.error!==void 0?t.error:n.error,location:n.location,revalidation:t.revalidation||n.revalidation}}componentDidCatch(t,n){console.error("React Router caught the following error during render",t,n)}render(){return this.state.error!==void 0?I.createElement(vn.Provider,{value:this.props.routeContext},I.createElement(Nh.Provider,{value:this.state.error,children:this.props.component})):this.props.children}}function c0(e){let{routeContext:t,match:n,children:r}=e,i=I.useContext(bl);return i&&i.static&&i.staticContext&&(n.route.errorElement||n.route.ErrorBoundary)&&(i.staticContext._deepestRenderedBoundaryId=n.route.id),I.createElement(vn.Provider,{value:t},r)}function d0(e,t,n,r){var i;if(t===void 0&&(t=[]),n===void 0&&(n=null),r===void 0&&(r=null),e==null){var o;if((o=n)!=null&&o.errors)e=n.matches;else return null}let s=e,a=(i=n)==null?void 0:i.errors;if(a!=null){let d=s.findIndex(f=>f.route.id&&(a==null?void 0:a[f.route.id])!==void 0);d>=0||Y(!1),s=s.slice(0,Math.min(s.length,d+1))}let l=!1,u=-1;if(n&&r&&r.v7_partialHydration)for(let d=0;d<s.length;d++){let f=s[d];if((f.route.HydrateFallback||f.route.hydrateFallbackElement)&&(u=d),f.route.id){let{loaderData:m,errors:g}=n,y=f.route.loader&&m[f.route.id]===void 0&&(!g||g[f.route.id]===void 0);if(f.route.lazy||y){l=!0,u>=0?s=s.slice(0,u+1):s=[s[0]];break}}}return s.reduceRight((d,f,m)=>{let g,y=!1,w=null,E=null;n&&(g=a&&f.route.id?a[f.route.id]:void 0,w=f.route.errorElement||l0,l&&(u<0&&m===0?(y=!0,E=null):u===m&&(y=!0,E=f.route.hydrateFallbackElement||null)));let h=t.concat(s.slice(0,m+1)),c=()=>{let p;return g?p=w:y?p=E:f.route.Component?p=I.createElement(f.route.Component,null):f.route.element?p=f.route.element:p=d,I.createElement(c0,{match:f,routeContext:{outlet:d,matches:h,isDataRoute:n!=null},children:p})};return n&&(f.route.ErrorBoundary||f.route.errorElement||m===0)?I.createElement(u0,{location:n.location,revalidation:n.revalidation,component:w,error:g,children:c(),routeContext:{outlet:null,matches:h,isDataRoute:!0}}):c()},null)}var xh=function(e){return e.UseBlocker="useBlocker",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e}(xh||{}),go=function(e){return e.UseBlocker="useBlocker",e.UseLoaderData="useLoaderData",e.UseActionData="useActionData",e.UseRouteError="useRouteError",e.UseNavigation="useNavigation",e.UseRouteLoaderData="useRouteLoaderData",e.UseMatches="useMatches",e.UseRevalidator="useRevalidator",e.UseNavigateStable="useNavigate",e.UseRouteId="useRouteId",e}(go||{});function f0(e){let t=I.useContext(bl);return t||Y(!1),t}function h0(e){let t=I.useContext(n0);return t||Y(!1),t}function p0(e){let t=I.useContext(vn);return t||Y(!1),t}function Lh(e){let t=p0(),n=t.matches[t.matches.length-1];return n.route.id||Y(!1),n.route.id}function m0(){var e;let t=I.useContext(Nh),n=h0(go.UseRouteError),r=Lh(go.UseRouteError);return t!==void 0?t:(e=n.errors)==null?void 0:e[r]}function g0(){let{router:e}=f0(xh.UseNavigateStable),t=Lh(go.UseNavigateStable),n=I.useRef(!1);return Oh(()=>{n.current=!0}),I.useCallback(function(i,o){o===void 0&&(o={}),n.current&&(typeof i=="number"?e.navigate(i):e.navigate(i,Kr({fromRouteId:t},o)))},[e,t])}function gr(e){Y(!1)}function v0(e){let{basename:t="/",children:n=null,location:r,navigationType:i=Nt.Pop,navigator:o,static:s=!1,future:a}=e;ti()&&Y(!1);let l=t.replace(/^\/*/,"/"),u=I.useMemo(()=>({basename:l,navigator:o,static:s,future:Kr({v7_relativeSplatPath:!1},a)}),[l,a,o,s]);typeof r=="string"&&(r=er(r));let{pathname:d="/",search:f="",hash:m="",state:g=null,key:y="default"}=r,w=I.useMemo(()=>{let E=zl(d,l);return E==null?null:{location:{pathname:E,search:f,hash:m,state:g,key:y},navigationType:i}},[l,d,f,m,g,y,i]);return w==null?null:I.createElement(gn.Provider,{value:u},I.createElement(Vo.Provider,{children:n,value:w}))}function y0(e){let{children:t,location:n}=e;return o0(xa(t),n)}new Promise(()=>{});function xa(e,t){t===void 0&&(t=[]);let n=[];return I.Children.forEach(e,(r,i)=>{if(!I.isValidElement(r))return;let o=[...t,i];if(r.type===I.Fragment){n.push.apply(n,xa(r.props.children,o));return}r.type!==gr&&Y(!1),!r.props.index||!r.props.children||Y(!1);let s={id:r.props.id||o.join("-"),caseSensitive:r.props.caseSensitive,element:r.props.element,Component:r.props.Component,index:r.props.index,path:r.props.path,loader:r.props.loader,action:r.props.action,errorElement:r.props.errorElement,ErrorBoundary:r.props.ErrorBoundary,hasErrorBoundary:r.props.ErrorBoundary!=null||r.props.errorElement!=null,shouldRevalidate:r.props.shouldRevalidate,handle:r.props.handle,lazy:r.props.lazy};r.props.children&&(s.children=xa(r.props.children,o)),n.push(s)}),n}/**
 * React Router DOM v6.23.1
 *
 * Copyright (c) Remix Software Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE.md file in the root directory of this source tree.
 *
 * @license MIT
 */function La(){return La=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},La.apply(this,arguments)}function w0(e,t){if(e==null)return{};var n={},r=Object.keys(e),i,o;for(o=0;o<r.length;o++)i=r[o],!(t.indexOf(i)>=0)&&(n[i]=e[i]);return n}function _0(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}function S0(e,t){return e.button===0&&(!t||t==="_self")&&!_0(e)}const E0=["onClick","relative","reloadDocument","replace","state","target","to","preventScrollReset","unstable_viewTransition"],I0="6";try{window.__reactRouterVersion=I0}catch{}const k0="startTransition",_c=wm[k0];function C0(e){let{basename:t,children:n,future:r,window:i}=e,o=I.useRef();o.current==null&&(o.current=Av({window:i,v5Compat:!0}));let s=o.current,[a,l]=I.useState({action:s.action,location:s.location}),{v7_startTransition:u}=r||{},d=I.useCallback(f=>{u&&_c?_c(()=>l(f)):l(f)},[l,u]);return I.useLayoutEffect(()=>s.listen(d),[s,d]),I.createElement(v0,{basename:t,children:n,location:a.location,navigationType:a.action,navigator:s,future:r})}const T0=typeof window<"u"&&typeof window.document<"u"&&typeof window.document.createElement<"u",P0=/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i,Da=I.forwardRef(function(t,n){let{onClick:r,relative:i,reloadDocument:o,replace:s,state:a,target:l,to:u,preventScrollReset:d,unstable_viewTransition:f}=t,m=w0(t,E0),{basename:g}=I.useContext(gn),y,w=!1;if(typeof u=="string"&&P0.test(u)&&(y=u,T0))try{let p=new URL(window.location.href),v=u.startsWith("//")?new URL(p.protocol+u):new URL(u),S=zl(v.pathname,g);v.origin===p.origin&&S!=null?u=S+v.search+v.hash:w=!0}catch{}let E=r0(u,{relative:i}),h=R0(u,{replace:s,state:a,target:l,preventScrollReset:d,relative:i,unstable_viewTransition:f});function c(p){r&&r(p),p.defaultPrevented||h(p)}return I.createElement("a",La({},m,{href:y||E,onClick:w||o?r:c,ref:n,target:l}))});var Sc;(function(e){e.UseScrollRestoration="useScrollRestoration",e.UseSubmit="useSubmit",e.UseSubmitFetcher="useSubmitFetcher",e.UseFetcher="useFetcher",e.useViewTransitionState="useViewTransitionState"})(Sc||(Sc={}));var Ec;(function(e){e.UseFetcher="useFetcher",e.UseFetchers="useFetchers",e.UseScrollRestoration="useScrollRestoration"})(Ec||(Ec={}));function R0(e,t){let{target:n,replace:r,state:i,preventScrollReset:o,relative:s,unstable_viewTransition:a}=t===void 0?{}:t,l=$l(),u=Ho(),d=Ah(e,{relative:s});return I.useCallback(f=>{if(S0(f,n)){f.preventDefault();let m=r!==void 0?r:mo(u)===mo(d);l(e,{replace:m,state:i,preventScrollReset:o,relative:s,unstable_viewTransition:a})}},[u,l,d,r,i,n,e,o,s,a])}var Ic={};/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Dh=function(e){const t=[];let n=0;for(let r=0;r<e.length;r++){let i=e.charCodeAt(r);i<128?t[n++]=i:i<2048?(t[n++]=i>>6|192,t[n++]=i&63|128):(i&64512)===55296&&r+1<e.length&&(e.charCodeAt(r+1)&64512)===56320?(i=65536+((i&1023)<<10)+(e.charCodeAt(++r)&1023),t[n++]=i>>18|240,t[n++]=i>>12&63|128,t[n++]=i>>6&63|128,t[n++]=i&63|128):(t[n++]=i>>12|224,t[n++]=i>>6&63|128,t[n++]=i&63|128)}return t},N0=function(e){const t=[];let n=0,r=0;for(;n<e.length;){const i=e[n++];if(i<128)t[r++]=String.fromCharCode(i);else if(i>191&&i<224){const o=e[n++];t[r++]=String.fromCharCode((i&31)<<6|o&63)}else if(i>239&&i<365){const o=e[n++],s=e[n++],a=e[n++],l=((i&7)<<18|(o&63)<<12|(s&63)<<6|a&63)-65536;t[r++]=String.fromCharCode(55296+(l>>10)),t[r++]=String.fromCharCode(56320+(l&1023))}else{const o=e[n++],s=e[n++];t[r++]=String.fromCharCode((i&15)<<12|(o&63)<<6|s&63)}}return t.join("")},Mh={byteToCharMap_:null,charToByteMap_:null,byteToCharMapWebSafe_:null,charToByteMapWebSafe_:null,ENCODED_VALS_BASE:"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",get ENCODED_VALS(){return this.ENCODED_VALS_BASE+"+/="},get ENCODED_VALS_WEBSAFE(){return this.ENCODED_VALS_BASE+"-_."},HAS_NATIVE_SUPPORT:typeof atob=="function",encodeByteArray(e,t){if(!Array.isArray(e))throw Error("encodeByteArray takes an array as a parameter");this.init_();const n=t?this.byteToCharMapWebSafe_:this.byteToCharMap_,r=[];for(let i=0;i<e.length;i+=3){const o=e[i],s=i+1<e.length,a=s?e[i+1]:0,l=i+2<e.length,u=l?e[i+2]:0,d=o>>2,f=(o&3)<<4|a>>4;let m=(a&15)<<2|u>>6,g=u&63;l||(g=64,s||(m=64)),r.push(n[d],n[f],n[m],n[g])}return r.join("")},encodeString(e,t){return this.HAS_NATIVE_SUPPORT&&!t?btoa(e):this.encodeByteArray(Dh(e),t)},decodeString(e,t){return this.HAS_NATIVE_SUPPORT&&!t?atob(e):N0(this.decodeStringToByteArray(e,t))},decodeStringToByteArray(e,t){this.init_();const n=t?this.charToByteMapWebSafe_:this.charToByteMap_,r=[];for(let i=0;i<e.length;){const o=n[e.charAt(i++)],a=i<e.length?n[e.charAt(i)]:0;++i;const u=i<e.length?n[e.charAt(i)]:64;++i;const f=i<e.length?n[e.charAt(i)]:64;if(++i,o==null||a==null||u==null||f==null)throw new O0;const m=o<<2|a>>4;if(r.push(m),u!==64){const g=a<<4&240|u>>2;if(r.push(g),f!==64){const y=u<<6&192|f;r.push(y)}}}return r},init_(){if(!this.byteToCharMap_){this.byteToCharMap_={},this.charToByteMap_={},this.byteToCharMapWebSafe_={},this.charToByteMapWebSafe_={};for(let e=0;e<this.ENCODED_VALS.length;e++)this.byteToCharMap_[e]=this.ENCODED_VALS.charAt(e),this.charToByteMap_[this.byteToCharMap_[e]]=e,this.byteToCharMapWebSafe_[e]=this.ENCODED_VALS_WEBSAFE.charAt(e),this.charToByteMapWebSafe_[this.byteToCharMapWebSafe_[e]]=e,e>=this.ENCODED_VALS_BASE.length&&(this.charToByteMap_[this.ENCODED_VALS_WEBSAFE.charAt(e)]=e,this.charToByteMapWebSafe_[this.ENCODED_VALS.charAt(e)]=e)}}};class O0 extends Error{constructor(){super(...arguments),this.name="DecodeBase64StringError"}}const A0=function(e){const t=Dh(e);return Mh.encodeByteArray(t,!0)},Uh=function(e){return A0(e).replace(/\./g,"")},jh=function(e){try{return Mh.decodeString(e,!0)}catch(t){console.error("base64Decode failed: ",t)}return null};/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function x0(){if(typeof self<"u")return self;if(typeof window<"u")return window;if(typeof global<"u")return global;throw new Error("Unable to locate global object.")}/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const L0=()=>x0().__FIREBASE_DEFAULTS__,D0=()=>{if(typeof process>"u"||typeof Ic>"u")return;const e=Ic.__FIREBASE_DEFAULTS__;if(e)return JSON.parse(e)},M0=()=>{if(typeof document>"u")return;let e;try{e=document.cookie.match(/__FIREBASE_DEFAULTS__=([^;]+)/)}catch{return}const t=e&&jh(e[1]);return t&&JSON.parse(t)},Bl=()=>{try{return L0()||D0()||M0()}catch(e){console.info(`Unable to get __FIREBASE_DEFAULTS__ due to: ${e}`);return}},U0=e=>{var t,n;return(n=(t=Bl())===null||t===void 0?void 0:t.emulatorHosts)===null||n===void 0?void 0:n[e]},Fh=()=>{var e;return(e=Bl())===null||e===void 0?void 0:e.config},zh=e=>{var t;return(t=Bl())===null||t===void 0?void 0:t[`_${e}`]};/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class j0{constructor(){this.reject=()=>{},this.resolve=()=>{},this.promise=new Promise((t,n)=>{this.resolve=t,this.reject=n})}wrapCallback(t){return(n,r)=>{n?this.reject(n):this.resolve(r),typeof t=="function"&&(this.promise.catch(()=>{}),t.length===1?t(n):t(n,r))}}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function ue(){return typeof navigator<"u"&&typeof navigator.userAgent=="string"?navigator.userAgent:""}function F0(){return typeof window<"u"&&!!(window.cordova||window.phonegap||window.PhoneGap)&&/ios|iphone|ipod|ipad|android|blackberry|iemobile/i.test(ue())}function bh(){const e=typeof chrome=="object"?chrome.runtime:typeof browser=="object"?browser.runtime:void 0;return typeof e=="object"&&e.id!==void 0}function z0(){return typeof navigator=="object"&&navigator.product==="ReactNative"}function b0(){const e=ue();return e.indexOf("MSIE ")>=0||e.indexOf("Trident/")>=0}function $h(){try{return typeof indexedDB=="object"}catch{return!1}}function Bh(){return new Promise((e,t)=>{try{let n=!0;const r="validate-browser-context-for-indexeddb-analytics-module",i=self.indexedDB.open(r);i.onsuccess=()=>{i.result.close(),n||self.indexedDB.deleteDatabase(r),e(!0)},i.onupgradeneeded=()=>{n=!1},i.onerror=()=>{var o;t(((o=i.error)===null||o===void 0?void 0:o.message)||"")}}catch(n){t(n)}})}function $0(){return!(typeof navigator>"u"||!navigator.cookieEnabled)}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const B0="FirebaseError";class Xe extends Error{constructor(t,n,r){super(n),this.code=t,this.customData=r,this.name=B0,Object.setPrototypeOf(this,Xe.prototype),Error.captureStackTrace&&Error.captureStackTrace(this,yn.prototype.create)}}class yn{constructor(t,n,r){this.service=t,this.serviceName=n,this.errors=r}create(t,...n){const r=n[0]||{},i=`${this.service}/${t}`,o=this.errors[t],s=o?V0(o,r):"Error",a=`${this.serviceName}: ${s} (${i}).`;return new Xe(i,a,r)}}function V0(e,t){return e.replace(H0,(n,r)=>{const i=t[r];return i!=null?String(i):`<${r}?>`})}const H0=/\{\$([^}]+)}/g;function W0(e){for(const t in e)if(Object.prototype.hasOwnProperty.call(e,t))return!1;return!0}function Gr(e,t){if(e===t)return!0;const n=Object.keys(e),r=Object.keys(t);for(const i of n){if(!r.includes(i))return!1;const o=e[i],s=t[i];if(kc(o)&&kc(s)){if(!Gr(o,s))return!1}else if(o!==s)return!1}for(const i of r)if(!n.includes(i))return!1;return!0}function kc(e){return e!==null&&typeof e=="object"}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function ni(e){const t=[];for(const[n,r]of Object.entries(e))Array.isArray(r)?r.forEach(i=>{t.push(encodeURIComponent(n)+"="+encodeURIComponent(i))}):t.push(encodeURIComponent(n)+"="+encodeURIComponent(r));return t.length?"&"+t.join("&"):""}function K0(e,t){const n=new G0(e,t);return n.subscribe.bind(n)}class G0{constructor(t,n){this.observers=[],this.unsubscribes=[],this.observerCount=0,this.task=Promise.resolve(),this.finalized=!1,this.onNoObservers=n,this.task.then(()=>{t(this)}).catch(r=>{this.error(r)})}next(t){this.forEachObserver(n=>{n.next(t)})}error(t){this.forEachObserver(n=>{n.error(t)}),this.close(t)}complete(){this.forEachObserver(t=>{t.complete()}),this.close()}subscribe(t,n,r){let i;if(t===void 0&&n===void 0&&r===void 0)throw new Error("Missing Observer.");q0(t,["next","error","complete"])?i=t:i={next:t,error:n,complete:r},i.next===void 0&&(i.next=Rs),i.error===void 0&&(i.error=Rs),i.complete===void 0&&(i.complete=Rs);const o=this.unsubscribeOne.bind(this,this.observers.length);return this.finalized&&this.task.then(()=>{try{this.finalError?i.error(this.finalError):i.complete()}catch{}}),this.observers.push(i),o}unsubscribeOne(t){this.observers===void 0||this.observers[t]===void 0||(delete this.observers[t],this.observerCount-=1,this.observerCount===0&&this.onNoObservers!==void 0&&this.onNoObservers(this))}forEachObserver(t){if(!this.finalized)for(let n=0;n<this.observers.length;n++)this.sendOne(n,t)}sendOne(t,n){this.task.then(()=>{if(this.observers!==void 0&&this.observers[t]!==void 0)try{n(this.observers[t])}catch(r){typeof console<"u"&&console.error&&console.error(r)}})}close(t){this.finalized||(this.finalized=!0,t!==void 0&&(this.finalError=t),this.task.then(()=>{this.observers=void 0,this.onNoObservers=void 0}))}}function q0(e,t){if(typeof e!="object"||e===null)return!1;for(const n of t)if(n in e&&typeof e[n]=="function")return!0;return!1}function Rs(){}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Q0=1e3,Y0=2,J0=4*60*60*1e3,X0=.5;function Cc(e,t=Q0,n=Y0){const r=t*Math.pow(n,e),i=Math.round(X0*r*(Math.random()-.5)*2);return Math.min(J0,r+i)}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function gt(e){return e&&e._delegate?e._delegate:e}class Ye{constructor(t,n,r){this.name=t,this.instanceFactory=n,this.type=r,this.multipleInstances=!1,this.serviceProps={},this.instantiationMode="LAZY",this.onInstanceCreated=null}setInstantiationMode(t){return this.instantiationMode=t,this}setMultipleInstances(t){return this.multipleInstances=t,this}setServiceProps(t){return this.serviceProps=t,this}setInstanceCreatedCallback(t){return this.onInstanceCreated=t,this}}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Yt="[DEFAULT]";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Z0{constructor(t,n){this.name=t,this.container=n,this.component=null,this.instances=new Map,this.instancesDeferred=new Map,this.instancesOptions=new Map,this.onInitCallbacks=new Map}get(t){const n=this.normalizeInstanceIdentifier(t);if(!this.instancesDeferred.has(n)){const r=new j0;if(this.instancesDeferred.set(n,r),this.isInitialized(n)||this.shouldAutoInitialize())try{const i=this.getOrInitializeService({instanceIdentifier:n});i&&r.resolve(i)}catch{}}return this.instancesDeferred.get(n).promise}getImmediate(t){var n;const r=this.normalizeInstanceIdentifier(t==null?void 0:t.identifier),i=(n=t==null?void 0:t.optional)!==null&&n!==void 0?n:!1;if(this.isInitialized(r)||this.shouldAutoInitialize())try{return this.getOrInitializeService({instanceIdentifier:r})}catch(o){if(i)return null;throw o}else{if(i)return null;throw Error(`Service ${this.name} is not available`)}}getComponent(){return this.component}setComponent(t){if(t.name!==this.name)throw Error(`Mismatching Component ${t.name} for Provider ${this.name}.`);if(this.component)throw Error(`Component for ${this.name} has already been provided`);if(this.component=t,!!this.shouldAutoInitialize()){if(ty(t))try{this.getOrInitializeService({instanceIdentifier:Yt})}catch{}for(const[n,r]of this.instancesDeferred.entries()){const i=this.normalizeInstanceIdentifier(n);try{const o=this.getOrInitializeService({instanceIdentifier:i});r.resolve(o)}catch{}}}}clearInstance(t=Yt){this.instancesDeferred.delete(t),this.instancesOptions.delete(t),this.instances.delete(t)}async delete(){const t=Array.from(this.instances.values());await Promise.all([...t.filter(n=>"INTERNAL"in n).map(n=>n.INTERNAL.delete()),...t.filter(n=>"_delete"in n).map(n=>n._delete())])}isComponentSet(){return this.component!=null}isInitialized(t=Yt){return this.instances.has(t)}getOptions(t=Yt){return this.instancesOptions.get(t)||{}}initialize(t={}){const{options:n={}}=t,r=this.normalizeInstanceIdentifier(t.instanceIdentifier);if(this.isInitialized(r))throw Error(`${this.name}(${r}) has already been initialized`);if(!this.isComponentSet())throw Error(`Component ${this.name} has not been registered yet`);const i=this.getOrInitializeService({instanceIdentifier:r,options:n});for(const[o,s]of this.instancesDeferred.entries()){const a=this.normalizeInstanceIdentifier(o);r===a&&s.resolve(i)}return i}onInit(t,n){var r;const i=this.normalizeInstanceIdentifier(n),o=(r=this.onInitCallbacks.get(i))!==null&&r!==void 0?r:new Set;o.add(t),this.onInitCallbacks.set(i,o);const s=this.instances.get(i);return s&&t(s,i),()=>{o.delete(t)}}invokeOnInitCallbacks(t,n){const r=this.onInitCallbacks.get(n);if(r)for(const i of r)try{i(t,n)}catch{}}getOrInitializeService({instanceIdentifier:t,options:n={}}){let r=this.instances.get(t);if(!r&&this.component&&(r=this.component.instanceFactory(this.container,{instanceIdentifier:ey(t),options:n}),this.instances.set(t,r),this.instancesOptions.set(t,n),this.invokeOnInitCallbacks(r,t),this.component.onInstanceCreated))try{this.component.onInstanceCreated(this.container,t,r)}catch{}return r||null}normalizeInstanceIdentifier(t=Yt){return this.component?this.component.multipleInstances?t:Yt:t}shouldAutoInitialize(){return!!this.component&&this.component.instantiationMode!=="EXPLICIT"}}function ey(e){return e===Yt?void 0:e}function ty(e){return e.instantiationMode==="EAGER"}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class ny{constructor(t){this.name=t,this.providers=new Map}addComponent(t){const n=this.getProvider(t.name);if(n.isComponentSet())throw new Error(`Component ${t.name} has already been registered with ${this.name}`);n.setComponent(t)}addOrOverwriteComponent(t){this.getProvider(t.name).isComponentSet()&&this.providers.delete(t.name),this.addComponent(t)}getProvider(t){if(this.providers.has(t))return this.providers.get(t);const n=new Z0(t,this);return this.providers.set(t,n),n}getProviders(){return Array.from(this.providers.values())}}/**
 * @license
 * Copyright 2017 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */var j;(function(e){e[e.DEBUG=0]="DEBUG",e[e.VERBOSE=1]="VERBOSE",e[e.INFO=2]="INFO",e[e.WARN=3]="WARN",e[e.ERROR=4]="ERROR",e[e.SILENT=5]="SILENT"})(j||(j={}));const ry={debug:j.DEBUG,verbose:j.VERBOSE,info:j.INFO,warn:j.WARN,error:j.ERROR,silent:j.SILENT},iy=j.INFO,oy={[j.DEBUG]:"log",[j.VERBOSE]:"log",[j.INFO]:"info",[j.WARN]:"warn",[j.ERROR]:"error"},sy=(e,t,...n)=>{if(t<e.logLevel)return;const r=new Date().toISOString(),i=oy[t];if(i)console[i](`[${r}]  ${e.name}:`,...n);else throw new Error(`Attempted to log a message with an invalid logType (value: ${t})`)};class Vl{constructor(t){this.name=t,this._logLevel=iy,this._logHandler=sy,this._userLogHandler=null}get logLevel(){return this._logLevel}set logLevel(t){if(!(t in j))throw new TypeError(`Invalid value "${t}" assigned to \`logLevel\``);this._logLevel=t}setLogLevel(t){this._logLevel=typeof t=="string"?ry[t]:t}get logHandler(){return this._logHandler}set logHandler(t){if(typeof t!="function")throw new TypeError("Value assigned to `logHandler` must be a function");this._logHandler=t}get userLogHandler(){return this._userLogHandler}set userLogHandler(t){this._userLogHandler=t}debug(...t){this._userLogHandler&&this._userLogHandler(this,j.DEBUG,...t),this._logHandler(this,j.DEBUG,...t)}log(...t){this._userLogHandler&&this._userLogHandler(this,j.VERBOSE,...t),this._logHandler(this,j.VERBOSE,...t)}info(...t){this._userLogHandler&&this._userLogHandler(this,j.INFO,...t),this._logHandler(this,j.INFO,...t)}warn(...t){this._userLogHandler&&this._userLogHandler(this,j.WARN,...t),this._logHandler(this,j.WARN,...t)}error(...t){this._userLogHandler&&this._userLogHandler(this,j.ERROR,...t),this._logHandler(this,j.ERROR,...t)}}const ay=(e,t)=>t.some(n=>e instanceof n);let Tc,Pc;function ly(){return Tc||(Tc=[IDBDatabase,IDBObjectStore,IDBIndex,IDBCursor,IDBTransaction])}function uy(){return Pc||(Pc=[IDBCursor.prototype.advance,IDBCursor.prototype.continue,IDBCursor.prototype.continuePrimaryKey])}const Vh=new WeakMap,Ma=new WeakMap,Hh=new WeakMap,Ns=new WeakMap,Hl=new WeakMap;function cy(e){const t=new Promise((n,r)=>{const i=()=>{e.removeEventListener("success",o),e.removeEventListener("error",s)},o=()=>{n(zt(e.result)),i()},s=()=>{r(e.error),i()};e.addEventListener("success",o),e.addEventListener("error",s)});return t.then(n=>{n instanceof IDBCursor&&Vh.set(n,e)}).catch(()=>{}),Hl.set(t,e),t}function dy(e){if(Ma.has(e))return;const t=new Promise((n,r)=>{const i=()=>{e.removeEventListener("complete",o),e.removeEventListener("error",s),e.removeEventListener("abort",s)},o=()=>{n(),i()},s=()=>{r(e.error||new DOMException("AbortError","AbortError")),i()};e.addEventListener("complete",o),e.addEventListener("error",s),e.addEventListener("abort",s)});Ma.set(e,t)}let Ua={get(e,t,n){if(e instanceof IDBTransaction){if(t==="done")return Ma.get(e);if(t==="objectStoreNames")return e.objectStoreNames||Hh.get(e);if(t==="store")return n.objectStoreNames[1]?void 0:n.objectStore(n.objectStoreNames[0])}return zt(e[t])},set(e,t,n){return e[t]=n,!0},has(e,t){return e instanceof IDBTransaction&&(t==="done"||t==="store")?!0:t in e}};function fy(e){Ua=e(Ua)}function hy(e){return e===IDBDatabase.prototype.transaction&&!("objectStoreNames"in IDBTransaction.prototype)?function(t,...n){const r=e.call(Os(this),t,...n);return Hh.set(r,t.sort?t.sort():[t]),zt(r)}:uy().includes(e)?function(...t){return e.apply(Os(this),t),zt(Vh.get(this))}:function(...t){return zt(e.apply(Os(this),t))}}function py(e){return typeof e=="function"?hy(e):(e instanceof IDBTransaction&&dy(e),ay(e,ly())?new Proxy(e,Ua):e)}function zt(e){if(e instanceof IDBRequest)return cy(e);if(Ns.has(e))return Ns.get(e);const t=py(e);return t!==e&&(Ns.set(e,t),Hl.set(t,e)),t}const Os=e=>Hl.get(e);function Wh(e,t,{blocked:n,upgrade:r,blocking:i,terminated:o}={}){const s=indexedDB.open(e,t),a=zt(s);return r&&s.addEventListener("upgradeneeded",l=>{r(zt(s.result),l.oldVersion,l.newVersion,zt(s.transaction),l)}),n&&s.addEventListener("blocked",l=>n(l.oldVersion,l.newVersion,l)),a.then(l=>{o&&l.addEventListener("close",()=>o()),i&&l.addEventListener("versionchange",u=>i(u.oldVersion,u.newVersion,u))}).catch(()=>{}),a}const my=["get","getKey","getAll","getAllKeys","count"],gy=["put","add","delete","clear"],As=new Map;function Rc(e,t){if(!(e instanceof IDBDatabase&&!(t in e)&&typeof t=="string"))return;if(As.get(t))return As.get(t);const n=t.replace(/FromIndex$/,""),r=t!==n,i=gy.includes(n);if(!(n in(r?IDBIndex:IDBObjectStore).prototype)||!(i||my.includes(n)))return;const o=async function(s,...a){const l=this.transaction(s,i?"readwrite":"readonly");let u=l.store;return r&&(u=u.index(a.shift())),(await Promise.all([u[n](...a),i&&l.done]))[0]};return As.set(t,o),o}fy(e=>({...e,get:(t,n,r)=>Rc(t,n)||e.get(t,n,r),has:(t,n)=>!!Rc(t,n)||e.has(t,n)}));/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class vy{constructor(t){this.container=t}getPlatformInfoString(){return this.container.getProviders().map(n=>{if(yy(n)){const r=n.getImmediate();return`${r.library}/${r.version}`}else return null}).filter(n=>n).join(" ")}}function yy(e){const t=e.getComponent();return(t==null?void 0:t.type)==="VERSION"}const ja="@firebase/app",Nc="0.10.5";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const un=new Vl("@firebase/app"),wy="@firebase/app-compat",_y="@firebase/analytics-compat",Sy="@firebase/analytics",Ey="@firebase/app-check-compat",Iy="@firebase/app-check",ky="@firebase/auth",Cy="@firebase/auth-compat",Ty="@firebase/database",Py="@firebase/database-compat",Ry="@firebase/functions",Ny="@firebase/functions-compat",Oy="@firebase/installations",Ay="@firebase/installations-compat",xy="@firebase/messaging",Ly="@firebase/messaging-compat",Dy="@firebase/performance",My="@firebase/performance-compat",Uy="@firebase/remote-config",jy="@firebase/remote-config-compat",Fy="@firebase/storage",zy="@firebase/storage-compat",by="@firebase/firestore",$y="@firebase/vertexai-preview",By="@firebase/firestore-compat",Vy="firebase",Hy="10.12.2";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Fa="[DEFAULT]",Wy={[ja]:"fire-core",[wy]:"fire-core-compat",[Sy]:"fire-analytics",[_y]:"fire-analytics-compat",[Iy]:"fire-app-check",[Ey]:"fire-app-check-compat",[ky]:"fire-auth",[Cy]:"fire-auth-compat",[Ty]:"fire-rtdb",[Py]:"fire-rtdb-compat",[Ry]:"fire-fn",[Ny]:"fire-fn-compat",[Oy]:"fire-iid",[Ay]:"fire-iid-compat",[xy]:"fire-fcm",[Ly]:"fire-fcm-compat",[Dy]:"fire-perf",[My]:"fire-perf-compat",[Uy]:"fire-rc",[jy]:"fire-rc-compat",[Fy]:"fire-gcs",[zy]:"fire-gcs-compat",[by]:"fire-fst",[By]:"fire-fst-compat",[$y]:"fire-vertex","fire-js":"fire-js",[Vy]:"fire-js-all"};/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const vo=new Map,Ky=new Map,za=new Map;function Oc(e,t){try{e.container.addComponent(t)}catch(n){un.debug(`Component ${t.name} failed to register with FirebaseApp ${e.name}`,n)}}function ht(e){const t=e.name;if(za.has(t))return un.debug(`There were multiple attempts to register component ${t}.`),!1;za.set(t,e);for(const n of vo.values())Oc(n,e);for(const n of Ky.values())Oc(n,e);return!0}function tr(e,t){const n=e.container.getProvider("heartbeat").getImmediate({optional:!0});return n&&n.triggerHeartbeat(),e.container.getProvider(t)}function He(e){return e.settings!==void 0}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Gy={"no-app":"No Firebase App '{$appName}' has been created - call initializeApp() first","bad-app-name":"Illegal App name: '{$appName}'","duplicate-app":"Firebase App named '{$appName}' already exists with different options or config","app-deleted":"Firebase App named '{$appName}' already deleted","server-app-deleted":"Firebase Server App has been deleted","no-options":"Need to provide options, when not being deployed to hosting via source.","invalid-app-argument":"firebase.{$appName}() takes either no argument or a Firebase App instance.","invalid-log-argument":"First argument to `onLog` must be null or a function.","idb-open":"Error thrown when opening IndexedDB. Original error: {$originalErrorMessage}.","idb-get":"Error thrown when reading from IndexedDB. Original error: {$originalErrorMessage}.","idb-set":"Error thrown when writing to IndexedDB. Original error: {$originalErrorMessage}.","idb-delete":"Error thrown when deleting from IndexedDB. Original error: {$originalErrorMessage}.","finalization-registry-not-supported":"FirebaseServerApp deleteOnDeref field defined but the JS runtime does not support FinalizationRegistry.","invalid-server-app-environment":"FirebaseServerApp is not for use in browser environments."},bt=new yn("app","Firebase",Gy);/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class qy{constructor(t,n,r){this._isDeleted=!1,this._options=Object.assign({},t),this._config=Object.assign({},n),this._name=n.name,this._automaticDataCollectionEnabled=n.automaticDataCollectionEnabled,this._container=r,this.container.addComponent(new Ye("app",()=>this,"PUBLIC"))}get automaticDataCollectionEnabled(){return this.checkDestroyed(),this._automaticDataCollectionEnabled}set automaticDataCollectionEnabled(t){this.checkDestroyed(),this._automaticDataCollectionEnabled=t}get name(){return this.checkDestroyed(),this._name}get options(){return this.checkDestroyed(),this._options}get config(){return this.checkDestroyed(),this._config}get container(){return this._container}get isDeleted(){return this._isDeleted}set isDeleted(t){this._isDeleted=t}checkDestroyed(){if(this.isDeleted)throw bt.create("app-deleted",{appName:this._name})}}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const ri=Hy;function Kh(e,t={}){let n=e;typeof t!="object"&&(t={name:t});const r=Object.assign({name:Fa,automaticDataCollectionEnabled:!1},t),i=r.name;if(typeof i!="string"||!i)throw bt.create("bad-app-name",{appName:String(i)});if(n||(n=Fh()),!n)throw bt.create("no-options");const o=vo.get(i);if(o){if(Gr(n,o.options)&&Gr(r,o.config))return o;throw bt.create("duplicate-app",{appName:i})}const s=new ny(i);for(const l of za.values())s.addComponent(l);const a=new qy(n,r,s);return vo.set(i,a),a}function Gh(e=Fa){const t=vo.get(e);if(!t&&e===Fa&&Fh())return Kh();if(!t)throw bt.create("no-app",{appName:e});return t}function Ge(e,t,n){var r;let i=(r=Wy[e])!==null&&r!==void 0?r:e;n&&(i+=`-${n}`);const o=i.match(/\s|\//),s=t.match(/\s|\//);if(o||s){const a=[`Unable to register library "${i}" with version "${t}":`];o&&a.push(`library name "${i}" contains illegal characters (whitespace or "/")`),o&&s&&a.push("and"),s&&a.push(`version name "${t}" contains illegal characters (whitespace or "/")`),un.warn(a.join(" "));return}ht(new Ye(`${i}-version`,()=>({library:i,version:t}),"VERSION"))}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Qy="firebase-heartbeat-database",Yy=1,qr="firebase-heartbeat-store";let xs=null;function qh(){return xs||(xs=Wh(Qy,Yy,{upgrade:(e,t)=>{switch(t){case 0:try{e.createObjectStore(qr)}catch(n){console.warn(n)}}}}).catch(e=>{throw bt.create("idb-open",{originalErrorMessage:e.message})})),xs}async function Jy(e){try{const n=(await qh()).transaction(qr),r=await n.objectStore(qr).get(Qh(e));return await n.done,r}catch(t){if(t instanceof Xe)un.warn(t.message);else{const n=bt.create("idb-get",{originalErrorMessage:t==null?void 0:t.message});un.warn(n.message)}}}async function Ac(e,t){try{const r=(await qh()).transaction(qr,"readwrite");await r.objectStore(qr).put(t,Qh(e)),await r.done}catch(n){if(n instanceof Xe)un.warn(n.message);else{const r=bt.create("idb-set",{originalErrorMessage:n==null?void 0:n.message});un.warn(r.message)}}}function Qh(e){return`${e.name}!${e.options.appId}`}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Xy=1024,Zy=30*24*60*60*1e3;class ew{constructor(t){this.container=t,this._heartbeatsCache=null;const n=this.container.getProvider("app").getImmediate();this._storage=new nw(n),this._heartbeatsCachePromise=this._storage.read().then(r=>(this._heartbeatsCache=r,r))}async triggerHeartbeat(){var t,n;const i=this.container.getProvider("platform-logger").getImmediate().getPlatformInfoString(),o=xc();if(!(((t=this._heartbeatsCache)===null||t===void 0?void 0:t.heartbeats)==null&&(this._heartbeatsCache=await this._heartbeatsCachePromise,((n=this._heartbeatsCache)===null||n===void 0?void 0:n.heartbeats)==null))&&!(this._heartbeatsCache.lastSentHeartbeatDate===o||this._heartbeatsCache.heartbeats.some(s=>s.date===o)))return this._heartbeatsCache.heartbeats.push({date:o,agent:i}),this._heartbeatsCache.heartbeats=this._heartbeatsCache.heartbeats.filter(s=>{const a=new Date(s.date).valueOf();return Date.now()-a<=Zy}),this._storage.overwrite(this._heartbeatsCache)}async getHeartbeatsHeader(){var t;if(this._heartbeatsCache===null&&await this._heartbeatsCachePromise,((t=this._heartbeatsCache)===null||t===void 0?void 0:t.heartbeats)==null||this._heartbeatsCache.heartbeats.length===0)return"";const n=xc(),{heartbeatsToSend:r,unsentEntries:i}=tw(this._heartbeatsCache.heartbeats),o=Uh(JSON.stringify({version:2,heartbeats:r}));return this._heartbeatsCache.lastSentHeartbeatDate=n,i.length>0?(this._heartbeatsCache.heartbeats=i,await this._storage.overwrite(this._heartbeatsCache)):(this._heartbeatsCache.heartbeats=[],this._storage.overwrite(this._heartbeatsCache)),o}}function xc(){return new Date().toISOString().substring(0,10)}function tw(e,t=Xy){const n=[];let r=e.slice();for(const i of e){const o=n.find(s=>s.agent===i.agent);if(o){if(o.dates.push(i.date),Lc(n)>t){o.dates.pop();break}}else if(n.push({agent:i.agent,dates:[i.date]}),Lc(n)>t){n.pop();break}r=r.slice(1)}return{heartbeatsToSend:n,unsentEntries:r}}class nw{constructor(t){this.app=t,this._canUseIndexedDBPromise=this.runIndexedDBEnvironmentCheck()}async runIndexedDBEnvironmentCheck(){return $h()?Bh().then(()=>!0).catch(()=>!1):!1}async read(){if(await this._canUseIndexedDBPromise){const n=await Jy(this.app);return n!=null&&n.heartbeats?n:{heartbeats:[]}}else return{heartbeats:[]}}async overwrite(t){var n;if(await this._canUseIndexedDBPromise){const i=await this.read();return Ac(this.app,{lastSentHeartbeatDate:(n=t.lastSentHeartbeatDate)!==null&&n!==void 0?n:i.lastSentHeartbeatDate,heartbeats:t.heartbeats})}else return}async add(t){var n;if(await this._canUseIndexedDBPromise){const i=await this.read();return Ac(this.app,{lastSentHeartbeatDate:(n=t.lastSentHeartbeatDate)!==null&&n!==void 0?n:i.lastSentHeartbeatDate,heartbeats:[...i.heartbeats,...t.heartbeats]})}else return}}function Lc(e){return Uh(JSON.stringify({version:2,heartbeats:e})).length}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function rw(e){ht(new Ye("platform-logger",t=>new vy(t),"PRIVATE")),ht(new Ye("heartbeat",t=>new ew(t),"PRIVATE")),Ge(ja,Nc,e),Ge(ja,Nc,"esm2017"),Ge("fire-js","")}rw("");var iw="firebase",ow="10.12.2";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */Ge(iw,ow,"app");const Yh="@firebase/installations",Wl="0.6.7";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Jh=1e4,Xh=`w:${Wl}`,Zh="FIS_v2",sw="https://firebaseinstallations.googleapis.com/v1",aw=60*60*1e3,lw="installations",uw="Installations";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const cw={"missing-app-config-values":'Missing App configuration value: "{$valueName}"',"not-registered":"Firebase Installation is not registered.","installation-not-found":"Firebase Installation not found.","request-failed":'{$requestName} request failed with error "{$serverCode} {$serverStatus}: {$serverMessage}"',"app-offline":"Could not process request. Application offline.","delete-pending-registration":"Can't delete installation while there is a pending registration request."},cn=new yn(lw,uw,cw);function ep(e){return e instanceof Xe&&e.code.includes("request-failed")}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function tp({projectId:e}){return`${sw}/projects/${e}/installations`}function np(e){return{token:e.token,requestStatus:2,expiresIn:fw(e.expiresIn),creationTime:Date.now()}}async function rp(e,t){const r=(await t.json()).error;return cn.create("request-failed",{requestName:e,serverCode:r.code,serverMessage:r.message,serverStatus:r.status})}function ip({apiKey:e}){return new Headers({"Content-Type":"application/json",Accept:"application/json","x-goog-api-key":e})}function dw(e,{refreshToken:t}){const n=ip(e);return n.append("Authorization",hw(t)),n}async function op(e){const t=await e();return t.status>=500&&t.status<600?e():t}function fw(e){return Number(e.replace("s","000"))}function hw(e){return`${Zh} ${e}`}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function pw({appConfig:e,heartbeatServiceProvider:t},{fid:n}){const r=tp(e),i=ip(e),o=t.getImmediate({optional:!0});if(o){const u=await o.getHeartbeatsHeader();u&&i.append("x-firebase-client",u)}const s={fid:n,authVersion:Zh,appId:e.appId,sdkVersion:Xh},a={method:"POST",headers:i,body:JSON.stringify(s)},l=await op(()=>fetch(r,a));if(l.ok){const u=await l.json();return{fid:u.fid||n,registrationStatus:2,refreshToken:u.refreshToken,authToken:np(u.authToken)}}else throw await rp("Create Installation",l)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function sp(e){return new Promise(t=>{setTimeout(t,e)})}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function mw(e){return btoa(String.fromCharCode(...e)).replace(/\+/g,"-").replace(/\//g,"_")}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const gw=/^[cdef][\w-]{21}$/,ba="";function vw(){try{const e=new Uint8Array(17);(self.crypto||self.msCrypto).getRandomValues(e),e[0]=112+e[0]%16;const n=yw(e);return gw.test(n)?n:ba}catch{return ba}}function yw(e){return mw(e).substr(0,22)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Wo(e){return`${e.appName}!${e.appId}`}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const ap=new Map;function lp(e,t){const n=Wo(e);up(n,t),ww(n,t)}function up(e,t){const n=ap.get(e);if(n)for(const r of n)r(t)}function ww(e,t){const n=_w();n&&n.postMessage({key:e,fid:t}),Sw()}let en=null;function _w(){return!en&&"BroadcastChannel"in self&&(en=new BroadcastChannel("[Firebase] FID Change"),en.onmessage=e=>{up(e.data.key,e.data.fid)}),en}function Sw(){ap.size===0&&en&&(en.close(),en=null)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Ew="firebase-installations-database",Iw=1,dn="firebase-installations-store";let Ls=null;function Kl(){return Ls||(Ls=Wh(Ew,Iw,{upgrade:(e,t)=>{switch(t){case 0:e.createObjectStore(dn)}}})),Ls}async function yo(e,t){const n=Wo(e),i=(await Kl()).transaction(dn,"readwrite"),o=i.objectStore(dn),s=await o.get(n);return await o.put(t,n),await i.done,(!s||s.fid!==t.fid)&&lp(e,t.fid),t}async function cp(e){const t=Wo(e),r=(await Kl()).transaction(dn,"readwrite");await r.objectStore(dn).delete(t),await r.done}async function Ko(e,t){const n=Wo(e),i=(await Kl()).transaction(dn,"readwrite"),o=i.objectStore(dn),s=await o.get(n),a=t(s);return a===void 0?await o.delete(n):await o.put(a,n),await i.done,a&&(!s||s.fid!==a.fid)&&lp(e,a.fid),a}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Gl(e){let t;const n=await Ko(e.appConfig,r=>{const i=kw(r),o=Cw(e,i);return t=o.registrationPromise,o.installationEntry});return n.fid===ba?{installationEntry:await t}:{installationEntry:n,registrationPromise:t}}function kw(e){const t=e||{fid:vw(),registrationStatus:0};return dp(t)}function Cw(e,t){if(t.registrationStatus===0){if(!navigator.onLine){const i=Promise.reject(cn.create("app-offline"));return{installationEntry:t,registrationPromise:i}}const n={fid:t.fid,registrationStatus:1,registrationTime:Date.now()},r=Tw(e,n);return{installationEntry:n,registrationPromise:r}}else return t.registrationStatus===1?{installationEntry:t,registrationPromise:Pw(e)}:{installationEntry:t}}async function Tw(e,t){try{const n=await pw(e,t);return yo(e.appConfig,n)}catch(n){throw ep(n)&&n.customData.serverCode===409?await cp(e.appConfig):await yo(e.appConfig,{fid:t.fid,registrationStatus:0}),n}}async function Pw(e){let t=await Dc(e.appConfig);for(;t.registrationStatus===1;)await sp(100),t=await Dc(e.appConfig);if(t.registrationStatus===0){const{installationEntry:n,registrationPromise:r}=await Gl(e);return r||n}return t}function Dc(e){return Ko(e,t=>{if(!t)throw cn.create("installation-not-found");return dp(t)})}function dp(e){return Rw(e)?{fid:e.fid,registrationStatus:0}:e}function Rw(e){return e.registrationStatus===1&&e.registrationTime+Jh<Date.now()}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Nw({appConfig:e,heartbeatServiceProvider:t},n){const r=Ow(e,n),i=dw(e,n),o=t.getImmediate({optional:!0});if(o){const u=await o.getHeartbeatsHeader();u&&i.append("x-firebase-client",u)}const s={installation:{sdkVersion:Xh,appId:e.appId}},a={method:"POST",headers:i,body:JSON.stringify(s)},l=await op(()=>fetch(r,a));if(l.ok){const u=await l.json();return np(u)}else throw await rp("Generate Auth Token",l)}function Ow(e,{fid:t}){return`${tp(e)}/${t}/authTokens:generate`}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function ql(e,t=!1){let n;const r=await Ko(e.appConfig,o=>{if(!fp(o))throw cn.create("not-registered");const s=o.authToken;if(!t&&Lw(s))return o;if(s.requestStatus===1)return n=Aw(e,t),o;{if(!navigator.onLine)throw cn.create("app-offline");const a=Mw(o);return n=xw(e,a),a}});return n?await n:r.authToken}async function Aw(e,t){let n=await Mc(e.appConfig);for(;n.authToken.requestStatus===1;)await sp(100),n=await Mc(e.appConfig);const r=n.authToken;return r.requestStatus===0?ql(e,t):r}function Mc(e){return Ko(e,t=>{if(!fp(t))throw cn.create("not-registered");const n=t.authToken;return Uw(n)?Object.assign(Object.assign({},t),{authToken:{requestStatus:0}}):t})}async function xw(e,t){try{const n=await Nw(e,t),r=Object.assign(Object.assign({},t),{authToken:n});return await yo(e.appConfig,r),n}catch(n){if(ep(n)&&(n.customData.serverCode===401||n.customData.serverCode===404))await cp(e.appConfig);else{const r=Object.assign(Object.assign({},t),{authToken:{requestStatus:0}});await yo(e.appConfig,r)}throw n}}function fp(e){return e!==void 0&&e.registrationStatus===2}function Lw(e){return e.requestStatus===2&&!Dw(e)}function Dw(e){const t=Date.now();return t<e.creationTime||e.creationTime+e.expiresIn<t+aw}function Mw(e){const t={requestStatus:1,requestTime:Date.now()};return Object.assign(Object.assign({},e),{authToken:t})}function Uw(e){return e.requestStatus===1&&e.requestTime+Jh<Date.now()}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function jw(e){const t=e,{installationEntry:n,registrationPromise:r}=await Gl(t);return r?r.catch(console.error):ql(t).catch(console.error),n.fid}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Fw(e,t=!1){const n=e;return await zw(n),(await ql(n,t)).token}async function zw(e){const{registrationPromise:t}=await Gl(e);t&&await t}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function bw(e){if(!e||!e.options)throw Ds("App Configuration");if(!e.name)throw Ds("App Name");const t=["projectId","apiKey","appId"];for(const n of t)if(!e.options[n])throw Ds(n);return{appName:e.name,projectId:e.options.projectId,apiKey:e.options.apiKey,appId:e.options.appId}}function Ds(e){return cn.create("missing-app-config-values",{valueName:e})}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const hp="installations",$w="installations-internal",Bw=e=>{const t=e.getProvider("app").getImmediate(),n=bw(t),r=tr(t,"heartbeat");return{app:t,appConfig:n,heartbeatServiceProvider:r,_delete:()=>Promise.resolve()}},Vw=e=>{const t=e.getProvider("app").getImmediate(),n=tr(t,hp).getImmediate();return{getId:()=>jw(n),getToken:i=>Fw(n,i)}};function Hw(){ht(new Ye(hp,Bw,"PUBLIC")),ht(new Ye($w,Vw,"PRIVATE"))}Hw();Ge(Yh,Wl);Ge(Yh,Wl,"esm2017");/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const wo="analytics",Ww="firebase_id",Kw="origin",Gw=60*1e3,qw="https://firebase.googleapis.com/v1alpha/projects/-/apps/{app-id}/webConfig",Ql="https://www.googletagmanager.com/gtag/js";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const we=new Vl("@firebase/analytics");/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Qw={"already-exists":"A Firebase Analytics instance with the appId {$id}  already exists. Only one Firebase Analytics instance can be created for each appId.","already-initialized":"initializeAnalytics() cannot be called again with different options than those it was initially called with. It can be called again with the same options to return the existing instance, or getAnalytics() can be used to get a reference to the already-intialized instance.","already-initialized-settings":"Firebase Analytics has already been initialized.settings() must be called before initializing any Analytics instanceor it will have no effect.","interop-component-reg-failed":"Firebase Analytics Interop Component failed to instantiate: {$reason}","invalid-analytics-context":"Firebase Analytics is not supported in this environment. Wrap initialization of analytics in analytics.isSupported() to prevent initialization in unsupported environments. Details: {$errorInfo}","indexeddb-unavailable":"IndexedDB unavailable or restricted in this environment. Wrap initialization of analytics in analytics.isSupported() to prevent initialization in unsupported environments. Details: {$errorInfo}","fetch-throttle":"The config fetch request timed out while in an exponential backoff state. Unix timestamp in milliseconds when fetch request throttling ends: {$throttleEndTimeMillis}.","config-fetch-failed":"Dynamic config fetch failed: [{$httpStatus}] {$responseMessage}","no-api-key":'The "apiKey" field is empty in the local Firebase config. Firebase Analytics requires this field tocontain a valid API key.',"no-app-id":'The "appId" field is empty in the local Firebase config. Firebase Analytics requires this field tocontain a valid app ID.',"no-client-id":'The "client_id" field is empty.',"invalid-gtag-resource":"Trusted Types detected an invalid gtag resource: {$gtagURL}."},Ie=new yn("analytics","Analytics",Qw);/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Yw(e){if(!e.startsWith(Ql)){const t=Ie.create("invalid-gtag-resource",{gtagURL:e});return we.warn(t.message),""}return e}function pp(e){return Promise.all(e.map(t=>t.catch(n=>n)))}function Jw(e,t){let n;return window.trustedTypes&&(n=window.trustedTypes.createPolicy(e,t)),n}function Xw(e,t){const n=Jw("firebase-js-sdk-policy",{createScriptURL:Yw}),r=document.createElement("script"),i=`${Ql}?l=${e}&id=${t}`;r.src=n?n==null?void 0:n.createScriptURL(i):i,r.async=!0,document.head.appendChild(r)}function Zw(e){let t=[];return Array.isArray(window[e])?t=window[e]:window[e]=t,t}async function e1(e,t,n,r,i,o){const s=r[i];try{if(s)await t[s];else{const l=(await pp(n)).find(u=>u.measurementId===i);l&&await t[l.appId]}}catch(a){we.error(a)}e("config",i,o)}async function t1(e,t,n,r,i){try{let o=[];if(i&&i.send_to){let s=i.send_to;Array.isArray(s)||(s=[s]);const a=await pp(n);for(const l of s){const u=a.find(f=>f.measurementId===l),d=u&&t[u.appId];if(d)o.push(d);else{o=[];break}}}o.length===0&&(o=Object.values(t)),await Promise.all(o),e("event",r,i||{})}catch(o){we.error(o)}}function n1(e,t,n,r){async function i(o,...s){try{if(o==="event"){const[a,l]=s;await t1(e,t,n,a,l)}else if(o==="config"){const[a,l]=s;await e1(e,t,n,r,a,l)}else if(o==="consent"){const[a,l]=s;e("consent",a,l)}else if(o==="get"){const[a,l,u]=s;e("get",a,l,u)}else if(o==="set"){const[a]=s;e("set",a)}else e(o,...s)}catch(a){we.error(a)}}return i}function r1(e,t,n,r,i){let o=function(...s){window[r].push(arguments)};return window[i]&&typeof window[i]=="function"&&(o=window[i]),window[i]=n1(o,e,t,n),{gtagCore:o,wrappedGtag:window[i]}}function i1(e){const t=window.document.getElementsByTagName("script");for(const n of Object.values(t))if(n.src&&n.src.includes(Ql)&&n.src.includes(e))return n;return null}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const o1=30,s1=1e3;class a1{constructor(t={},n=s1){this.throttleMetadata=t,this.intervalMillis=n}getThrottleMetadata(t){return this.throttleMetadata[t]}setThrottleMetadata(t,n){this.throttleMetadata[t]=n}deleteThrottleMetadata(t){delete this.throttleMetadata[t]}}const mp=new a1;function l1(e){return new Headers({Accept:"application/json","x-goog-api-key":e})}async function u1(e){var t;const{appId:n,apiKey:r}=e,i={method:"GET",headers:l1(r)},o=qw.replace("{app-id}",n),s=await fetch(o,i);if(s.status!==200&&s.status!==304){let a="";try{const l=await s.json();!((t=l.error)===null||t===void 0)&&t.message&&(a=l.error.message)}catch{}throw Ie.create("config-fetch-failed",{httpStatus:s.status,responseMessage:a})}return s.json()}async function c1(e,t=mp,n){const{appId:r,apiKey:i,measurementId:o}=e.options;if(!r)throw Ie.create("no-app-id");if(!i){if(o)return{measurementId:o,appId:r};throw Ie.create("no-api-key")}const s=t.getThrottleMetadata(r)||{backoffCount:0,throttleEndTimeMillis:Date.now()},a=new h1;return setTimeout(async()=>{a.abort()},Gw),gp({appId:r,apiKey:i,measurementId:o},s,a,t)}async function gp(e,{throttleEndTimeMillis:t,backoffCount:n},r,i=mp){var o;const{appId:s,measurementId:a}=e;try{await d1(r,t)}catch(l){if(a)return we.warn(`Timed out fetching this Firebase app's measurement ID from the server. Falling back to the measurement ID ${a} provided in the "measurementId" field in the local Firebase config. [${l==null?void 0:l.message}]`),{appId:s,measurementId:a};throw l}try{const l=await u1(e);return i.deleteThrottleMetadata(s),l}catch(l){const u=l;if(!f1(u)){if(i.deleteThrottleMetadata(s),a)return we.warn(`Failed to fetch this Firebase app's measurement ID from the server. Falling back to the measurement ID ${a} provided in the "measurementId" field in the local Firebase config. [${u==null?void 0:u.message}]`),{appId:s,measurementId:a};throw l}const d=Number((o=u==null?void 0:u.customData)===null||o===void 0?void 0:o.httpStatus)===503?Cc(n,i.intervalMillis,o1):Cc(n,i.intervalMillis),f={throttleEndTimeMillis:Date.now()+d,backoffCount:n+1};return i.setThrottleMetadata(s,f),we.debug(`Calling attemptFetch again in ${d} millis`),gp(e,f,r,i)}}function d1(e,t){return new Promise((n,r)=>{const i=Math.max(t-Date.now(),0),o=setTimeout(n,i);e.addEventListener(()=>{clearTimeout(o),r(Ie.create("fetch-throttle",{throttleEndTimeMillis:t}))})})}function f1(e){if(!(e instanceof Xe)||!e.customData)return!1;const t=Number(e.customData.httpStatus);return t===429||t===500||t===503||t===504}class h1{constructor(){this.listeners=[]}addEventListener(t){this.listeners.push(t)}abort(){this.listeners.forEach(t=>t())}}async function p1(e,t,n,r,i){if(i&&i.global){e("event",n,r);return}else{const o=await t,s=Object.assign(Object.assign({},r),{send_to:o});e("event",n,s)}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function m1(){if($h())try{await Bh()}catch(e){return we.warn(Ie.create("indexeddb-unavailable",{errorInfo:e==null?void 0:e.toString()}).message),!1}else return we.warn(Ie.create("indexeddb-unavailable",{errorInfo:"IndexedDB is not available in this environment."}).message),!1;return!0}async function g1(e,t,n,r,i,o,s){var a;const l=c1(e);l.then(g=>{n[g.measurementId]=g.appId,e.options.measurementId&&g.measurementId!==e.options.measurementId&&we.warn(`The measurement ID in the local Firebase config (${e.options.measurementId}) does not match the measurement ID fetched from the server (${g.measurementId}). To ensure analytics events are always sent to the correct Analytics property, update the measurement ID field in the local config or remove it from the local config.`)}).catch(g=>we.error(g)),t.push(l);const u=m1().then(g=>{if(g)return r.getId()}),[d,f]=await Promise.all([l,u]);i1(o)||Xw(o,d.measurementId),i("js",new Date);const m=(a=s==null?void 0:s.config)!==null&&a!==void 0?a:{};return m[Kw]="firebase",m.update=!0,f!=null&&(m[Ww]=f),i("config",d.measurementId,m),d.measurementId}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class v1{constructor(t){this.app=t}_delete(){return delete Tr[this.app.options.appId],Promise.resolve()}}let Tr={},Uc=[];const jc={};let Ms="dataLayer",y1="gtag",Fc,vp,zc=!1;function w1(){const e=[];if(bh()&&e.push("This is a browser extension environment."),$0()||e.push("Cookies are not available."),e.length>0){const t=e.map((r,i)=>`(${i+1}) ${r}`).join(" "),n=Ie.create("invalid-analytics-context",{errorInfo:t});we.warn(n.message)}}function _1(e,t,n){w1();const r=e.options.appId;if(!r)throw Ie.create("no-app-id");if(!e.options.apiKey)if(e.options.measurementId)we.warn(`The "apiKey" field is empty in the local Firebase config. This is needed to fetch the latest measurement ID for this Firebase app. Falling back to the measurement ID ${e.options.measurementId} provided in the "measurementId" field in the local Firebase config.`);else throw Ie.create("no-api-key");if(Tr[r]!=null)throw Ie.create("already-exists",{id:r});if(!zc){Zw(Ms);const{wrappedGtag:o,gtagCore:s}=r1(Tr,Uc,jc,Ms,y1);vp=o,Fc=s,zc=!0}return Tr[r]=g1(e,Uc,jc,t,Fc,Ms,n),new v1(e)}function S1(e=Gh()){e=gt(e);const t=tr(e,wo);return t.isInitialized()?t.getImmediate():E1(e)}function E1(e,t={}){const n=tr(e,wo);if(n.isInitialized()){const i=n.getImmediate();if(Gr(t,n.getOptions()))return i;throw Ie.create("already-initialized")}return n.initialize({options:t})}function I1(e,t,n,r){e=gt(e),p1(vp,Tr[e.app.options.appId],t,n,r).catch(i=>we.error(i))}const bc="@firebase/analytics",$c="0.10.4";function k1(){ht(new Ye(wo,(t,{options:n})=>{const r=t.getProvider("app").getImmediate(),i=t.getProvider("installations-internal").getImmediate();return _1(r,i,n)},"PUBLIC")),ht(new Ye("analytics-internal",e,"PRIVATE")),Ge(bc,$c),Ge(bc,$c,"esm2017");function e(t){try{const n=t.getProvider(wo).getImmediate();return{logEvent:(r,i,o)=>I1(n,r,i,o)}}catch(n){throw Ie.create("interop-component-reg-failed",{reason:n})}}}k1();function Yl(e,t){var n={};for(var r in e)Object.prototype.hasOwnProperty.call(e,r)&&t.indexOf(r)<0&&(n[r]=e[r]);if(e!=null&&typeof Object.getOwnPropertySymbols=="function")for(var i=0,r=Object.getOwnPropertySymbols(e);i<r.length;i++)t.indexOf(r[i])<0&&Object.prototype.propertyIsEnumerable.call(e,r[i])&&(n[r[i]]=e[r[i]]);return n}function yp(){return{"dependent-sdk-initialized-before-auth":"Another Firebase SDK was initialized and is trying to use Auth before Auth is initialized. Please be sure to call `initializeAuth` or `getAuth` before starting any other Firebase SDK."}}const C1=yp,wp=new yn("auth","Firebase",yp());/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const _o=new Vl("@firebase/auth");function T1(e,...t){_o.logLevel<=j.WARN&&_o.warn(`Auth (${ri}): ${e}`,...t)}function bi(e,...t){_o.logLevel<=j.ERROR&&_o.error(`Auth (${ri}): ${e}`,...t)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Je(e,...t){throw Xl(e,...t)}function qe(e,...t){return Xl(e,...t)}function Jl(e,t,n){const r=Object.assign(Object.assign({},C1()),{[t]:n});return new yn("auth","Firebase",r).create(t,{appName:e.name})}function lt(e){return Jl(e,"operation-not-supported-in-this-environment","Operations that alter the current user are not supported in conjunction with FirebaseServerApp")}function P1(e,t,n){const r=n;if(!(t instanceof r))throw r.name!==t.constructor.name&&Je(e,"argument-error"),Jl(e,"argument-error",`Type of ${t.constructor.name} does not match expected instance.Did you pass a reference from a different Auth SDK?`)}function Xl(e,...t){if(typeof e!="string"){const n=t[0],r=[...t.slice(1)];return r[0]&&(r[0].appName=e.name),e._errorFactory.create(n,...r)}return wp.create(e,...t)}function O(e,t,...n){if(!e)throw Xl(t,...n)}function rt(e){const t="INTERNAL ASSERTION FAILED: "+e;throw bi(t),new Error(t)}function pt(e,t){e||rt(t)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function $a(){var e;return typeof self<"u"&&((e=self.location)===null||e===void 0?void 0:e.href)||""}function R1(){return Bc()==="http:"||Bc()==="https:"}function Bc(){var e;return typeof self<"u"&&((e=self.location)===null||e===void 0?void 0:e.protocol)||null}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function N1(){return typeof navigator<"u"&&navigator&&"onLine"in navigator&&typeof navigator.onLine=="boolean"&&(R1()||bh()||"connection"in navigator)?navigator.onLine:!0}function O1(){if(typeof navigator>"u")return null;const e=navigator;return e.languages&&e.languages[0]||e.language||null}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class ii{constructor(t,n){this.shortDelay=t,this.longDelay=n,pt(n>t,"Short delay should be less than long delay!"),this.isMobile=F0()||z0()}get(){return N1()?this.isMobile?this.longDelay:this.shortDelay:Math.min(5e3,this.shortDelay)}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Zl(e,t){pt(e.emulator,"Emulator should always be set here");const{url:n}=e.emulator;return t?`${n}${t.startsWith("/")?t.slice(1):t}`:n}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class _p{static initialize(t,n,r){this.fetchImpl=t,n&&(this.headersImpl=n),r&&(this.responseImpl=r)}static fetch(){if(this.fetchImpl)return this.fetchImpl;if(typeof self<"u"&&"fetch"in self)return self.fetch;if(typeof globalThis<"u"&&globalThis.fetch)return globalThis.fetch;if(typeof fetch<"u")return fetch;rt("Could not find fetch implementation, make sure you call FetchProvider.initialize() with an appropriate polyfill")}static headers(){if(this.headersImpl)return this.headersImpl;if(typeof self<"u"&&"Headers"in self)return self.Headers;if(typeof globalThis<"u"&&globalThis.Headers)return globalThis.Headers;if(typeof Headers<"u")return Headers;rt("Could not find Headers implementation, make sure you call FetchProvider.initialize() with an appropriate polyfill")}static response(){if(this.responseImpl)return this.responseImpl;if(typeof self<"u"&&"Response"in self)return self.Response;if(typeof globalThis<"u"&&globalThis.Response)return globalThis.Response;if(typeof Response<"u")return Response;rt("Could not find Response implementation, make sure you call FetchProvider.initialize() with an appropriate polyfill")}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const A1={CREDENTIAL_MISMATCH:"custom-token-mismatch",MISSING_CUSTOM_TOKEN:"internal-error",INVALID_IDENTIFIER:"invalid-email",MISSING_CONTINUE_URI:"internal-error",INVALID_PASSWORD:"wrong-password",MISSING_PASSWORD:"missing-password",INVALID_LOGIN_CREDENTIALS:"invalid-credential",EMAIL_EXISTS:"email-already-in-use",PASSWORD_LOGIN_DISABLED:"operation-not-allowed",INVALID_IDP_RESPONSE:"invalid-credential",INVALID_PENDING_TOKEN:"invalid-credential",FEDERATED_USER_ID_ALREADY_LINKED:"credential-already-in-use",MISSING_REQ_TYPE:"internal-error",EMAIL_NOT_FOUND:"user-not-found",RESET_PASSWORD_EXCEED_LIMIT:"too-many-requests",EXPIRED_OOB_CODE:"expired-action-code",INVALID_OOB_CODE:"invalid-action-code",MISSING_OOB_CODE:"internal-error",CREDENTIAL_TOO_OLD_LOGIN_AGAIN:"requires-recent-login",INVALID_ID_TOKEN:"invalid-user-token",TOKEN_EXPIRED:"user-token-expired",USER_NOT_FOUND:"user-token-expired",TOO_MANY_ATTEMPTS_TRY_LATER:"too-many-requests",PASSWORD_DOES_NOT_MEET_REQUIREMENTS:"password-does-not-meet-requirements",INVALID_CODE:"invalid-verification-code",INVALID_SESSION_INFO:"invalid-verification-id",INVALID_TEMPORARY_PROOF:"invalid-credential",MISSING_SESSION_INFO:"missing-verification-id",SESSION_EXPIRED:"code-expired",MISSING_ANDROID_PACKAGE_NAME:"missing-android-pkg-name",UNAUTHORIZED_DOMAIN:"unauthorized-continue-uri",INVALID_OAUTH_CLIENT_ID:"invalid-oauth-client-id",ADMIN_ONLY_OPERATION:"admin-restricted-operation",INVALID_MFA_PENDING_CREDENTIAL:"invalid-multi-factor-session",MFA_ENROLLMENT_NOT_FOUND:"multi-factor-info-not-found",MISSING_MFA_ENROLLMENT_ID:"missing-multi-factor-info",MISSING_MFA_PENDING_CREDENTIAL:"missing-multi-factor-session",SECOND_FACTOR_EXISTS:"second-factor-already-in-use",SECOND_FACTOR_LIMIT_EXCEEDED:"maximum-second-factor-count-exceeded",BLOCKING_FUNCTION_ERROR_RESPONSE:"internal-error",RECAPTCHA_NOT_ENABLED:"recaptcha-not-enabled",MISSING_RECAPTCHA_TOKEN:"missing-recaptcha-token",INVALID_RECAPTCHA_TOKEN:"invalid-recaptcha-token",INVALID_RECAPTCHA_ACTION:"invalid-recaptcha-action",MISSING_CLIENT_TYPE:"missing-client-type",MISSING_RECAPTCHA_VERSION:"missing-recaptcha-version",INVALID_RECAPTCHA_VERSION:"invalid-recaptcha-version",INVALID_REQ_TYPE:"invalid-req-type"};/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const x1=new ii(3e4,6e4);function oi(e,t){return e.tenantId&&!t.tenantId?Object.assign(Object.assign({},t),{tenantId:e.tenantId}):t}async function wn(e,t,n,r,i={}){return Sp(e,i,async()=>{let o={},s={};r&&(t==="GET"?s=r:o={body:JSON.stringify(r)});const a=ni(Object.assign({key:e.config.apiKey},s)).slice(1),l=await e._getAdditionalHeaders();return l["Content-Type"]="application/json",e.languageCode&&(l["X-Firebase-Locale"]=e.languageCode),_p.fetch()(Ip(e,e.config.apiHost,n,a),Object.assign({method:t,headers:l,referrerPolicy:"no-referrer"},o))})}async function Sp(e,t,n){e._canInitEmulator=!1;const r=Object.assign(Object.assign({},A1),t);try{const i=new D1(e),o=await Promise.race([n(),i.promise]);i.clearNetworkTimeout();const s=await o.json();if("needConfirmation"in s)throw Pi(e,"account-exists-with-different-credential",s);if(o.ok&&!("errorMessage"in s))return s;{const a=o.ok?s.errorMessage:s.error.message,[l,u]=a.split(" : ");if(l==="FEDERATED_USER_ID_ALREADY_LINKED")throw Pi(e,"credential-already-in-use",s);if(l==="EMAIL_EXISTS")throw Pi(e,"email-already-in-use",s);if(l==="USER_DISABLED")throw Pi(e,"user-disabled",s);const d=r[l]||l.toLowerCase().replace(/[_\s]+/g,"-");if(u)throw Jl(e,d,u);Je(e,d)}}catch(i){if(i instanceof Xe)throw i;Je(e,"network-request-failed",{message:String(i)})}}async function Ep(e,t,n,r,i={}){const o=await wn(e,t,n,r,i);return"mfaPendingCredential"in o&&Je(e,"multi-factor-auth-required",{_serverResponse:o}),o}function Ip(e,t,n,r){const i=`${t}${n}?${r}`;return e.config.emulator?Zl(e.config,i):`${e.config.apiScheme}://${i}`}function L1(e){switch(e){case"ENFORCE":return"ENFORCE";case"AUDIT":return"AUDIT";case"OFF":return"OFF";default:return"ENFORCEMENT_STATE_UNSPECIFIED"}}class D1{constructor(t){this.auth=t,this.timer=null,this.promise=new Promise((n,r)=>{this.timer=setTimeout(()=>r(qe(this.auth,"network-request-failed")),x1.get())})}clearNetworkTimeout(){clearTimeout(this.timer)}}function Pi(e,t,n){const r={appName:e.name};n.email&&(r.email=n.email),n.phoneNumber&&(r.phoneNumber=n.phoneNumber);const i=qe(e,t,r);return i.customData._tokenResponse=n,i}function Vc(e){return e!==void 0&&e.enterprise!==void 0}class M1{constructor(t){if(this.siteKey="",this.recaptchaEnforcementState=[],t.recaptchaKey===void 0)throw new Error("recaptchaKey undefined");this.siteKey=t.recaptchaKey.split("/")[3],this.recaptchaEnforcementState=t.recaptchaEnforcementState}getProviderEnforcementState(t){if(!this.recaptchaEnforcementState||this.recaptchaEnforcementState.length===0)return null;for(const n of this.recaptchaEnforcementState)if(n.provider&&n.provider===t)return L1(n.enforcementState);return null}isProviderEnabled(t){return this.getProviderEnforcementState(t)==="ENFORCE"||this.getProviderEnforcementState(t)==="AUDIT"}}async function U1(e,t){return wn(e,"GET","/v2/recaptchaConfig",oi(e,t))}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function j1(e,t){return wn(e,"POST","/v1/accounts:delete",t)}async function kp(e,t){return wn(e,"POST","/v1/accounts:lookup",t)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Pr(e){if(e)try{const t=new Date(Number(e));if(!isNaN(t.getTime()))return t.toUTCString()}catch{}}async function F1(e,t=!1){const n=gt(e),r=await n.getIdToken(t),i=eu(r);O(i&&i.exp&&i.auth_time&&i.iat,n.auth,"internal-error");const o=typeof i.firebase=="object"?i.firebase:void 0,s=o==null?void 0:o.sign_in_provider;return{claims:i,token:r,authTime:Pr(Us(i.auth_time)),issuedAtTime:Pr(Us(i.iat)),expirationTime:Pr(Us(i.exp)),signInProvider:s||null,signInSecondFactor:(o==null?void 0:o.sign_in_second_factor)||null}}function Us(e){return Number(e)*1e3}function eu(e){const[t,n,r]=e.split(".");if(t===void 0||n===void 0||r===void 0)return bi("JWT malformed, contained fewer than 3 sections"),null;try{const i=jh(n);return i?JSON.parse(i):(bi("Failed to decode base64 JWT payload"),null)}catch(i){return bi("Caught error parsing JWT payload as JSON",i==null?void 0:i.toString()),null}}function Hc(e){const t=eu(e);return O(t,"internal-error"),O(typeof t.exp<"u","internal-error"),O(typeof t.iat<"u","internal-error"),Number(t.exp)-Number(t.iat)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Qr(e,t,n=!1){if(n)return t;try{return await t}catch(r){throw r instanceof Xe&&z1(r)&&e.auth.currentUser===e&&await e.auth.signOut(),r}}function z1({code:e}){return e==="auth/user-disabled"||e==="auth/user-token-expired"}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class b1{constructor(t){this.user=t,this.isRunning=!1,this.timerId=null,this.errorBackoff=3e4}_start(){this.isRunning||(this.isRunning=!0,this.schedule())}_stop(){this.isRunning&&(this.isRunning=!1,this.timerId!==null&&clearTimeout(this.timerId))}getInterval(t){var n;if(t){const r=this.errorBackoff;return this.errorBackoff=Math.min(this.errorBackoff*2,96e4),r}else{this.errorBackoff=3e4;const i=((n=this.user.stsTokenManager.expirationTime)!==null&&n!==void 0?n:0)-Date.now()-3e5;return Math.max(0,i)}}schedule(t=!1){if(!this.isRunning)return;const n=this.getInterval(t);this.timerId=setTimeout(async()=>{await this.iteration()},n)}async iteration(){try{await this.user.getIdToken(!0)}catch(t){(t==null?void 0:t.code)==="auth/network-request-failed"&&this.schedule(!0);return}this.schedule()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Ba{constructor(t,n){this.createdAt=t,this.lastLoginAt=n,this._initializeTime()}_initializeTime(){this.lastSignInTime=Pr(this.lastLoginAt),this.creationTime=Pr(this.createdAt)}_copy(t){this.createdAt=t.createdAt,this.lastLoginAt=t.lastLoginAt,this._initializeTime()}toJSON(){return{createdAt:this.createdAt,lastLoginAt:this.lastLoginAt}}}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function So(e){var t;const n=e.auth,r=await e.getIdToken(),i=await Qr(e,kp(n,{idToken:r}));O(i==null?void 0:i.users.length,n,"internal-error");const o=i.users[0];e._notifyReloadListener(o);const s=!((t=o.providerUserInfo)===null||t===void 0)&&t.length?Cp(o.providerUserInfo):[],a=B1(e.providerData,s),l=e.isAnonymous,u=!(e.email&&o.passwordHash)&&!(a!=null&&a.length),d=l?u:!1,f={uid:o.localId,displayName:o.displayName||null,photoURL:o.photoUrl||null,email:o.email||null,emailVerified:o.emailVerified||!1,phoneNumber:o.phoneNumber||null,tenantId:o.tenantId||null,providerData:a,metadata:new Ba(o.createdAt,o.lastLoginAt),isAnonymous:d};Object.assign(e,f)}async function $1(e){const t=gt(e);await So(t),await t.auth._persistUserIfCurrent(t),t.auth._notifyListenersIfCurrent(t)}function B1(e,t){return[...e.filter(r=>!t.some(i=>i.providerId===r.providerId)),...t]}function Cp(e){return e.map(t=>{var{providerId:n}=t,r=Yl(t,["providerId"]);return{providerId:n,uid:r.rawId||"",displayName:r.displayName||null,email:r.email||null,phoneNumber:r.phoneNumber||null,photoURL:r.photoUrl||null}})}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function V1(e,t){const n=await Sp(e,{},async()=>{const r=ni({grant_type:"refresh_token",refresh_token:t}).slice(1),{tokenApiHost:i,apiKey:o}=e.config,s=Ip(e,i,"/v1/token",`key=${o}`),a=await e._getAdditionalHeaders();return a["Content-Type"]="application/x-www-form-urlencoded",_p.fetch()(s,{method:"POST",headers:a,body:r})});return{accessToken:n.access_token,expiresIn:n.expires_in,refreshToken:n.refresh_token}}async function H1(e,t){return wn(e,"POST","/v2/accounts:revokeToken",oi(e,t))}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class $n{constructor(){this.refreshToken=null,this.accessToken=null,this.expirationTime=null}get isExpired(){return!this.expirationTime||Date.now()>this.expirationTime-3e4}updateFromServerResponse(t){O(t.idToken,"internal-error"),O(typeof t.idToken<"u","internal-error"),O(typeof t.refreshToken<"u","internal-error");const n="expiresIn"in t&&typeof t.expiresIn<"u"?Number(t.expiresIn):Hc(t.idToken);this.updateTokensAndExpiration(t.idToken,t.refreshToken,n)}updateFromIdToken(t){O(t.length!==0,"internal-error");const n=Hc(t);this.updateTokensAndExpiration(t,null,n)}async getToken(t,n=!1){return!n&&this.accessToken&&!this.isExpired?this.accessToken:(O(this.refreshToken,t,"user-token-expired"),this.refreshToken?(await this.refresh(t,this.refreshToken),this.accessToken):null)}clearRefreshToken(){this.refreshToken=null}async refresh(t,n){const{accessToken:r,refreshToken:i,expiresIn:o}=await V1(t,n);this.updateTokensAndExpiration(r,i,Number(o))}updateTokensAndExpiration(t,n,r){this.refreshToken=n||null,this.accessToken=t||null,this.expirationTime=Date.now()+r*1e3}static fromJSON(t,n){const{refreshToken:r,accessToken:i,expirationTime:o}=n,s=new $n;return r&&(O(typeof r=="string","internal-error",{appName:t}),s.refreshToken=r),i&&(O(typeof i=="string","internal-error",{appName:t}),s.accessToken=i),o&&(O(typeof o=="number","internal-error",{appName:t}),s.expirationTime=o),s}toJSON(){return{refreshToken:this.refreshToken,accessToken:this.accessToken,expirationTime:this.expirationTime}}_assign(t){this.accessToken=t.accessToken,this.refreshToken=t.refreshToken,this.expirationTime=t.expirationTime}_clone(){return Object.assign(new $n,this.toJSON())}_performRefresh(){return rt("not implemented")}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function yt(e,t){O(typeof e=="string"||typeof e>"u","internal-error",{appName:t})}class it{constructor(t){var{uid:n,auth:r,stsTokenManager:i}=t,o=Yl(t,["uid","auth","stsTokenManager"]);this.providerId="firebase",this.proactiveRefresh=new b1(this),this.reloadUserInfo=null,this.reloadListener=null,this.uid=n,this.auth=r,this.stsTokenManager=i,this.accessToken=i.accessToken,this.displayName=o.displayName||null,this.email=o.email||null,this.emailVerified=o.emailVerified||!1,this.phoneNumber=o.phoneNumber||null,this.photoURL=o.photoURL||null,this.isAnonymous=o.isAnonymous||!1,this.tenantId=o.tenantId||null,this.providerData=o.providerData?[...o.providerData]:[],this.metadata=new Ba(o.createdAt||void 0,o.lastLoginAt||void 0)}async getIdToken(t){const n=await Qr(this,this.stsTokenManager.getToken(this.auth,t));return O(n,this.auth,"internal-error"),this.accessToken!==n&&(this.accessToken=n,await this.auth._persistUserIfCurrent(this),this.auth._notifyListenersIfCurrent(this)),n}getIdTokenResult(t){return F1(this,t)}reload(){return $1(this)}_assign(t){this!==t&&(O(this.uid===t.uid,this.auth,"internal-error"),this.displayName=t.displayName,this.photoURL=t.photoURL,this.email=t.email,this.emailVerified=t.emailVerified,this.phoneNumber=t.phoneNumber,this.isAnonymous=t.isAnonymous,this.tenantId=t.tenantId,this.providerData=t.providerData.map(n=>Object.assign({},n)),this.metadata._copy(t.metadata),this.stsTokenManager._assign(t.stsTokenManager))}_clone(t){const n=new it(Object.assign(Object.assign({},this),{auth:t,stsTokenManager:this.stsTokenManager._clone()}));return n.metadata._copy(this.metadata),n}_onReload(t){O(!this.reloadListener,this.auth,"internal-error"),this.reloadListener=t,this.reloadUserInfo&&(this._notifyReloadListener(this.reloadUserInfo),this.reloadUserInfo=null)}_notifyReloadListener(t){this.reloadListener?this.reloadListener(t):this.reloadUserInfo=t}_startProactiveRefresh(){this.proactiveRefresh._start()}_stopProactiveRefresh(){this.proactiveRefresh._stop()}async _updateTokensIfNecessary(t,n=!1){let r=!1;t.idToken&&t.idToken!==this.stsTokenManager.accessToken&&(this.stsTokenManager.updateFromServerResponse(t),r=!0),n&&await So(this),await this.auth._persistUserIfCurrent(this),r&&this.auth._notifyListenersIfCurrent(this)}async delete(){if(He(this.auth.app))return Promise.reject(lt(this.auth));const t=await this.getIdToken();return await Qr(this,j1(this.auth,{idToken:t})),this.stsTokenManager.clearRefreshToken(),this.auth.signOut()}toJSON(){return Object.assign(Object.assign({uid:this.uid,email:this.email||void 0,emailVerified:this.emailVerified,displayName:this.displayName||void 0,isAnonymous:this.isAnonymous,photoURL:this.photoURL||void 0,phoneNumber:this.phoneNumber||void 0,tenantId:this.tenantId||void 0,providerData:this.providerData.map(t=>Object.assign({},t)),stsTokenManager:this.stsTokenManager.toJSON(),_redirectEventId:this._redirectEventId},this.metadata.toJSON()),{apiKey:this.auth.config.apiKey,appName:this.auth.name})}get refreshToken(){return this.stsTokenManager.refreshToken||""}static _fromJSON(t,n){var r,i,o,s,a,l,u,d;const f=(r=n.displayName)!==null&&r!==void 0?r:void 0,m=(i=n.email)!==null&&i!==void 0?i:void 0,g=(o=n.phoneNumber)!==null&&o!==void 0?o:void 0,y=(s=n.photoURL)!==null&&s!==void 0?s:void 0,w=(a=n.tenantId)!==null&&a!==void 0?a:void 0,E=(l=n._redirectEventId)!==null&&l!==void 0?l:void 0,h=(u=n.createdAt)!==null&&u!==void 0?u:void 0,c=(d=n.lastLoginAt)!==null&&d!==void 0?d:void 0,{uid:p,emailVerified:v,isAnonymous:S,providerData:P,stsTokenManager:R}=n;O(p&&R,t,"internal-error");const N=$n.fromJSON(this.name,R);O(typeof p=="string",t,"internal-error"),yt(f,t.name),yt(m,t.name),O(typeof v=="boolean",t,"internal-error"),O(typeof S=="boolean",t,"internal-error"),yt(g,t.name),yt(y,t.name),yt(w,t.name),yt(E,t.name),yt(h,t.name),yt(c,t.name);const z=new it({uid:p,auth:t,email:m,emailVerified:v,displayName:f,isAnonymous:S,photoURL:y,phoneNumber:g,tenantId:w,stsTokenManager:N,createdAt:h,lastLoginAt:c});return P&&Array.isArray(P)&&(z.providerData=P.map(L=>Object.assign({},L))),E&&(z._redirectEventId=E),z}static async _fromIdTokenResponse(t,n,r=!1){const i=new $n;i.updateFromServerResponse(n);const o=new it({uid:n.localId,auth:t,stsTokenManager:i,isAnonymous:r});return await So(o),o}static async _fromGetAccountInfoResponse(t,n,r){const i=n.users[0];O(i.localId!==void 0,"internal-error");const o=i.providerUserInfo!==void 0?Cp(i.providerUserInfo):[],s=!(i.email&&i.passwordHash)&&!(o!=null&&o.length),a=new $n;a.updateFromIdToken(r);const l=new it({uid:i.localId,auth:t,stsTokenManager:a,isAnonymous:s}),u={uid:i.localId,displayName:i.displayName||null,photoURL:i.photoUrl||null,email:i.email||null,emailVerified:i.emailVerified||!1,phoneNumber:i.phoneNumber||null,tenantId:i.tenantId||null,providerData:o,metadata:new Ba(i.createdAt,i.lastLoginAt),isAnonymous:!(i.email&&i.passwordHash)&&!(o!=null&&o.length)};return Object.assign(l,u),l}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Wc=new Map;function ot(e){pt(e instanceof Function,"Expected a class definition");let t=Wc.get(e);return t?(pt(t instanceof e,"Instance stored in cache mismatched with class"),t):(t=new e,Wc.set(e,t),t)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Tp{constructor(){this.type="NONE",this.storage={}}async _isAvailable(){return!0}async _set(t,n){this.storage[t]=n}async _get(t){const n=this.storage[t];return n===void 0?null:n}async _remove(t){delete this.storage[t]}_addListener(t,n){}_removeListener(t,n){}}Tp.type="NONE";const Kc=Tp;/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function $i(e,t,n){return`firebase:${e}:${t}:${n}`}class Bn{constructor(t,n,r){this.persistence=t,this.auth=n,this.userKey=r;const{config:i,name:o}=this.auth;this.fullUserKey=$i(this.userKey,i.apiKey,o),this.fullPersistenceKey=$i("persistence",i.apiKey,o),this.boundEventHandler=n._onStorageEvent.bind(n),this.persistence._addListener(this.fullUserKey,this.boundEventHandler)}setCurrentUser(t){return this.persistence._set(this.fullUserKey,t.toJSON())}async getCurrentUser(){const t=await this.persistence._get(this.fullUserKey);return t?it._fromJSON(this.auth,t):null}removeCurrentUser(){return this.persistence._remove(this.fullUserKey)}savePersistenceForRedirect(){return this.persistence._set(this.fullPersistenceKey,this.persistence.type)}async setPersistence(t){if(this.persistence===t)return;const n=await this.getCurrentUser();if(await this.removeCurrentUser(),this.persistence=t,n)return this.setCurrentUser(n)}delete(){this.persistence._removeListener(this.fullUserKey,this.boundEventHandler)}static async create(t,n,r="authUser"){if(!n.length)return new Bn(ot(Kc),t,r);const i=(await Promise.all(n.map(async u=>{if(await u._isAvailable())return u}))).filter(u=>u);let o=i[0]||ot(Kc);const s=$i(r,t.config.apiKey,t.name);let a=null;for(const u of n)try{const d=await u._get(s);if(d){const f=it._fromJSON(t,d);u!==o&&(a=f),o=u;break}}catch{}const l=i.filter(u=>u._shouldAllowMigration);return!o._shouldAllowMigration||!l.length?new Bn(o,t,r):(o=l[0],a&&await o._set(s,a.toJSON()),await Promise.all(n.map(async u=>{if(u!==o)try{await u._remove(s)}catch{}})),new Bn(o,t,r))}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Gc(e){const t=e.toLowerCase();if(t.includes("opera/")||t.includes("opr/")||t.includes("opios/"))return"Opera";if(Np(t))return"IEMobile";if(t.includes("msie")||t.includes("trident/"))return"IE";if(t.includes("edge/"))return"Edge";if(Pp(t))return"Firefox";if(t.includes("silk/"))return"Silk";if(Ap(t))return"Blackberry";if(xp(t))return"Webos";if(tu(t))return"Safari";if((t.includes("chrome/")||Rp(t))&&!t.includes("edge/"))return"Chrome";if(Op(t))return"Android";{const n=/([a-zA-Z\d\.]+)\/[a-zA-Z\d\.]*$/,r=e.match(n);if((r==null?void 0:r.length)===2)return r[1]}return"Other"}function Pp(e=ue()){return/firefox\//i.test(e)}function tu(e=ue()){const t=e.toLowerCase();return t.includes("safari/")&&!t.includes("chrome/")&&!t.includes("crios/")&&!t.includes("android")}function Rp(e=ue()){return/crios\//i.test(e)}function Np(e=ue()){return/iemobile/i.test(e)}function Op(e=ue()){return/android/i.test(e)}function Ap(e=ue()){return/blackberry/i.test(e)}function xp(e=ue()){return/webos/i.test(e)}function Go(e=ue()){return/iphone|ipad|ipod/i.test(e)||/macintosh/i.test(e)&&/mobile/i.test(e)}function W1(e=ue()){var t;return Go(e)&&!!(!((t=window.navigator)===null||t===void 0)&&t.standalone)}function K1(){return b0()&&document.documentMode===10}function Lp(e=ue()){return Go(e)||Op(e)||xp(e)||Ap(e)||/windows phone/i.test(e)||Np(e)}function G1(){try{return!!(window&&window!==window.top)}catch{return!1}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Dp(e,t=[]){let n;switch(e){case"Browser":n=Gc(ue());break;case"Worker":n=`${Gc(ue())}-${e}`;break;default:n=e}const r=t.length?t.join(","):"FirebaseCore-web";return`${n}/JsCore/${ri}/${r}`}/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class q1{constructor(t){this.auth=t,this.queue=[]}pushCallback(t,n){const r=o=>new Promise((s,a)=>{try{const l=t(o);s(l)}catch(l){a(l)}});r.onAbort=n,this.queue.push(r);const i=this.queue.length-1;return()=>{this.queue[i]=()=>Promise.resolve()}}async runMiddleware(t){if(this.auth.currentUser===t)return;const n=[];try{for(const r of this.queue)await r(t),r.onAbort&&n.push(r.onAbort)}catch(r){n.reverse();for(const i of n)try{i()}catch{}throw this.auth._errorFactory.create("login-blocked",{originalMessage:r==null?void 0:r.message})}}}/**
 * @license
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Q1(e,t={}){return wn(e,"GET","/v2/passwordPolicy",oi(e,t))}/**
 * @license
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Y1=6;class J1{constructor(t){var n,r,i,o;const s=t.customStrengthOptions;this.customStrengthOptions={},this.customStrengthOptions.minPasswordLength=(n=s.minPasswordLength)!==null&&n!==void 0?n:Y1,s.maxPasswordLength&&(this.customStrengthOptions.maxPasswordLength=s.maxPasswordLength),s.containsLowercaseCharacter!==void 0&&(this.customStrengthOptions.containsLowercaseLetter=s.containsLowercaseCharacter),s.containsUppercaseCharacter!==void 0&&(this.customStrengthOptions.containsUppercaseLetter=s.containsUppercaseCharacter),s.containsNumericCharacter!==void 0&&(this.customStrengthOptions.containsNumericCharacter=s.containsNumericCharacter),s.containsNonAlphanumericCharacter!==void 0&&(this.customStrengthOptions.containsNonAlphanumericCharacter=s.containsNonAlphanumericCharacter),this.enforcementState=t.enforcementState,this.enforcementState==="ENFORCEMENT_STATE_UNSPECIFIED"&&(this.enforcementState="OFF"),this.allowedNonAlphanumericCharacters=(i=(r=t.allowedNonAlphanumericCharacters)===null||r===void 0?void 0:r.join(""))!==null&&i!==void 0?i:"",this.forceUpgradeOnSignin=(o=t.forceUpgradeOnSignin)!==null&&o!==void 0?o:!1,this.schemaVersion=t.schemaVersion}validatePassword(t){var n,r,i,o,s,a;const l={isValid:!0,passwordPolicy:this};return this.validatePasswordLengthOptions(t,l),this.validatePasswordCharacterOptions(t,l),l.isValid&&(l.isValid=(n=l.meetsMinPasswordLength)!==null&&n!==void 0?n:!0),l.isValid&&(l.isValid=(r=l.meetsMaxPasswordLength)!==null&&r!==void 0?r:!0),l.isValid&&(l.isValid=(i=l.containsLowercaseLetter)!==null&&i!==void 0?i:!0),l.isValid&&(l.isValid=(o=l.containsUppercaseLetter)!==null&&o!==void 0?o:!0),l.isValid&&(l.isValid=(s=l.containsNumericCharacter)!==null&&s!==void 0?s:!0),l.isValid&&(l.isValid=(a=l.containsNonAlphanumericCharacter)!==null&&a!==void 0?a:!0),l}validatePasswordLengthOptions(t,n){const r=this.customStrengthOptions.minPasswordLength,i=this.customStrengthOptions.maxPasswordLength;r&&(n.meetsMinPasswordLength=t.length>=r),i&&(n.meetsMaxPasswordLength=t.length<=i)}validatePasswordCharacterOptions(t,n){this.updatePasswordCharacterOptionsStatuses(n,!1,!1,!1,!1);let r;for(let i=0;i<t.length;i++)r=t.charAt(i),this.updatePasswordCharacterOptionsStatuses(n,r>="a"&&r<="z",r>="A"&&r<="Z",r>="0"&&r<="9",this.allowedNonAlphanumericCharacters.includes(r))}updatePasswordCharacterOptionsStatuses(t,n,r,i,o){this.customStrengthOptions.containsLowercaseLetter&&(t.containsLowercaseLetter||(t.containsLowercaseLetter=n)),this.customStrengthOptions.containsUppercaseLetter&&(t.containsUppercaseLetter||(t.containsUppercaseLetter=r)),this.customStrengthOptions.containsNumericCharacter&&(t.containsNumericCharacter||(t.containsNumericCharacter=i)),this.customStrengthOptions.containsNonAlphanumericCharacter&&(t.containsNonAlphanumericCharacter||(t.containsNonAlphanumericCharacter=o))}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class X1{constructor(t,n,r,i){this.app=t,this.heartbeatServiceProvider=n,this.appCheckServiceProvider=r,this.config=i,this.currentUser=null,this.emulatorConfig=null,this.operations=Promise.resolve(),this.authStateSubscription=new qc(this),this.idTokenSubscription=new qc(this),this.beforeStateQueue=new q1(this),this.redirectUser=null,this.isProactiveRefreshEnabled=!1,this.EXPECTED_PASSWORD_POLICY_SCHEMA_VERSION=1,this._canInitEmulator=!0,this._isInitialized=!1,this._deleted=!1,this._initializationPromise=null,this._popupRedirectResolver=null,this._errorFactory=wp,this._agentRecaptchaConfig=null,this._tenantRecaptchaConfigs={},this._projectPasswordPolicy=null,this._tenantPasswordPolicies={},this.lastNotifiedUid=void 0,this.languageCode=null,this.tenantId=null,this.settings={appVerificationDisabledForTesting:!1},this.frameworks=[],this.name=t.name,this.clientVersion=i.sdkClientVersion}_initializeWithPersistence(t,n){return n&&(this._popupRedirectResolver=ot(n)),this._initializationPromise=this.queue(async()=>{var r,i;if(!this._deleted&&(this.persistenceManager=await Bn.create(this,t),!this._deleted)){if(!((r=this._popupRedirectResolver)===null||r===void 0)&&r._shouldInitProactively)try{await this._popupRedirectResolver._initialize(this)}catch{}await this.initializeCurrentUser(n),this.lastNotifiedUid=((i=this.currentUser)===null||i===void 0?void 0:i.uid)||null,!this._deleted&&(this._isInitialized=!0)}}),this._initializationPromise}async _onStorageEvent(){if(this._deleted)return;const t=await this.assertedPersistence.getCurrentUser();if(!(!this.currentUser&&!t)){if(this.currentUser&&t&&this.currentUser.uid===t.uid){this._currentUser._assign(t),await this.currentUser.getIdToken();return}await this._updateCurrentUser(t,!0)}}async initializeCurrentUserFromIdToken(t){try{const n=await kp(this,{idToken:t}),r=await it._fromGetAccountInfoResponse(this,n,t);await this.directlySetCurrentUser(r)}catch(n){console.warn("FirebaseServerApp could not login user with provided authIdToken: ",n),await this.directlySetCurrentUser(null)}}async initializeCurrentUser(t){var n;if(He(this.app)){const s=this.app.settings.authIdToken;return s?new Promise(a=>{setTimeout(()=>this.initializeCurrentUserFromIdToken(s).then(a,a))}):this.directlySetCurrentUser(null)}const r=await this.assertedPersistence.getCurrentUser();let i=r,o=!1;if(t&&this.config.authDomain){await this.getOrInitRedirectPersistenceManager();const s=(n=this.redirectUser)===null||n===void 0?void 0:n._redirectEventId,a=i==null?void 0:i._redirectEventId,l=await this.tryRedirectSignIn(t);(!s||s===a)&&(l!=null&&l.user)&&(i=l.user,o=!0)}if(!i)return this.directlySetCurrentUser(null);if(!i._redirectEventId){if(o)try{await this.beforeStateQueue.runMiddleware(i)}catch(s){i=r,this._popupRedirectResolver._overrideRedirectResult(this,()=>Promise.reject(s))}return i?this.reloadAndSetCurrentUserOrClear(i):this.directlySetCurrentUser(null)}return O(this._popupRedirectResolver,this,"argument-error"),await this.getOrInitRedirectPersistenceManager(),this.redirectUser&&this.redirectUser._redirectEventId===i._redirectEventId?this.directlySetCurrentUser(i):this.reloadAndSetCurrentUserOrClear(i)}async tryRedirectSignIn(t){let n=null;try{n=await this._popupRedirectResolver._completeRedirectFn(this,t,!0)}catch{await this._setRedirectUser(null)}return n}async reloadAndSetCurrentUserOrClear(t){try{await So(t)}catch(n){if((n==null?void 0:n.code)!=="auth/network-request-failed")return this.directlySetCurrentUser(null)}return this.directlySetCurrentUser(t)}useDeviceLanguage(){this.languageCode=O1()}async _delete(){this._deleted=!0}async updateCurrentUser(t){if(He(this.app))return Promise.reject(lt(this));const n=t?gt(t):null;return n&&O(n.auth.config.apiKey===this.config.apiKey,this,"invalid-user-token"),this._updateCurrentUser(n&&n._clone(this))}async _updateCurrentUser(t,n=!1){if(!this._deleted)return t&&O(this.tenantId===t.tenantId,this,"tenant-id-mismatch"),n||await this.beforeStateQueue.runMiddleware(t),this.queue(async()=>{await this.directlySetCurrentUser(t),this.notifyAuthListeners()})}async signOut(){return He(this.app)?Promise.reject(lt(this)):(await this.beforeStateQueue.runMiddleware(null),(this.redirectPersistenceManager||this._popupRedirectResolver)&&await this._setRedirectUser(null),this._updateCurrentUser(null,!0))}setPersistence(t){return He(this.app)?Promise.reject(lt(this)):this.queue(async()=>{await this.assertedPersistence.setPersistence(ot(t))})}_getRecaptchaConfig(){return this.tenantId==null?this._agentRecaptchaConfig:this._tenantRecaptchaConfigs[this.tenantId]}async validatePassword(t){this._getPasswordPolicyInternal()||await this._updatePasswordPolicy();const n=this._getPasswordPolicyInternal();return n.schemaVersion!==this.EXPECTED_PASSWORD_POLICY_SCHEMA_VERSION?Promise.reject(this._errorFactory.create("unsupported-password-policy-schema-version",{})):n.validatePassword(t)}_getPasswordPolicyInternal(){return this.tenantId===null?this._projectPasswordPolicy:this._tenantPasswordPolicies[this.tenantId]}async _updatePasswordPolicy(){const t=await Q1(this),n=new J1(t);this.tenantId===null?this._projectPasswordPolicy=n:this._tenantPasswordPolicies[this.tenantId]=n}_getPersistence(){return this.assertedPersistence.persistence.type}_updateErrorMap(t){this._errorFactory=new yn("auth","Firebase",t())}onAuthStateChanged(t,n,r){return this.registerStateListener(this.authStateSubscription,t,n,r)}beforeAuthStateChanged(t,n){return this.beforeStateQueue.pushCallback(t,n)}onIdTokenChanged(t,n,r){return this.registerStateListener(this.idTokenSubscription,t,n,r)}authStateReady(){return new Promise((t,n)=>{if(this.currentUser)t();else{const r=this.onAuthStateChanged(()=>{r(),t()},n)}})}async revokeAccessToken(t){if(this.currentUser){const n=await this.currentUser.getIdToken(),r={providerId:"apple.com",tokenType:"ACCESS_TOKEN",token:t,idToken:n};this.tenantId!=null&&(r.tenantId=this.tenantId),await H1(this,r)}}toJSON(){var t;return{apiKey:this.config.apiKey,authDomain:this.config.authDomain,appName:this.name,currentUser:(t=this._currentUser)===null||t===void 0?void 0:t.toJSON()}}async _setRedirectUser(t,n){const r=await this.getOrInitRedirectPersistenceManager(n);return t===null?r.removeCurrentUser():r.setCurrentUser(t)}async getOrInitRedirectPersistenceManager(t){if(!this.redirectPersistenceManager){const n=t&&ot(t)||this._popupRedirectResolver;O(n,this,"argument-error"),this.redirectPersistenceManager=await Bn.create(this,[ot(n._redirectPersistence)],"redirectUser"),this.redirectUser=await this.redirectPersistenceManager.getCurrentUser()}return this.redirectPersistenceManager}async _redirectUserForId(t){var n,r;return this._isInitialized&&await this.queue(async()=>{}),((n=this._currentUser)===null||n===void 0?void 0:n._redirectEventId)===t?this._currentUser:((r=this.redirectUser)===null||r===void 0?void 0:r._redirectEventId)===t?this.redirectUser:null}async _persistUserIfCurrent(t){if(t===this.currentUser)return this.queue(async()=>this.directlySetCurrentUser(t))}_notifyListenersIfCurrent(t){t===this.currentUser&&this.notifyAuthListeners()}_key(){return`${this.config.authDomain}:${this.config.apiKey}:${this.name}`}_startProactiveRefresh(){this.isProactiveRefreshEnabled=!0,this.currentUser&&this._currentUser._startProactiveRefresh()}_stopProactiveRefresh(){this.isProactiveRefreshEnabled=!1,this.currentUser&&this._currentUser._stopProactiveRefresh()}get _currentUser(){return this.currentUser}notifyAuthListeners(){var t,n;if(!this._isInitialized)return;this.idTokenSubscription.next(this.currentUser);const r=(n=(t=this.currentUser)===null||t===void 0?void 0:t.uid)!==null&&n!==void 0?n:null;this.lastNotifiedUid!==r&&(this.lastNotifiedUid=r,this.authStateSubscription.next(this.currentUser))}registerStateListener(t,n,r,i){if(this._deleted)return()=>{};const o=typeof n=="function"?n:n.next.bind(n);let s=!1;const a=this._isInitialized?Promise.resolve():this._initializationPromise;if(O(a,this,"internal-error"),a.then(()=>{s||o(this.currentUser)}),typeof n=="function"){const l=t.addObserver(n,r,i);return()=>{s=!0,l()}}else{const l=t.addObserver(n);return()=>{s=!0,l()}}}async directlySetCurrentUser(t){this.currentUser&&this.currentUser!==t&&this._currentUser._stopProactiveRefresh(),t&&this.isProactiveRefreshEnabled&&t._startProactiveRefresh(),this.currentUser=t,t?await this.assertedPersistence.setCurrentUser(t):await this.assertedPersistence.removeCurrentUser()}queue(t){return this.operations=this.operations.then(t,t),this.operations}get assertedPersistence(){return O(this.persistenceManager,this,"internal-error"),this.persistenceManager}_logFramework(t){!t||this.frameworks.includes(t)||(this.frameworks.push(t),this.frameworks.sort(),this.clientVersion=Dp(this.config.clientPlatform,this._getFrameworks()))}_getFrameworks(){return this.frameworks}async _getAdditionalHeaders(){var t;const n={"X-Client-Version":this.clientVersion};this.app.options.appId&&(n["X-Firebase-gmpid"]=this.app.options.appId);const r=await((t=this.heartbeatServiceProvider.getImmediate({optional:!0}))===null||t===void 0?void 0:t.getHeartbeatsHeader());r&&(n["X-Firebase-Client"]=r);const i=await this._getAppCheckToken();return i&&(n["X-Firebase-AppCheck"]=i),n}async _getAppCheckToken(){var t;const n=await((t=this.appCheckServiceProvider.getImmediate({optional:!0}))===null||t===void 0?void 0:t.getToken());return n!=null&&n.error&&T1(`Error while retrieving App Check token: ${n.error}`),n==null?void 0:n.token}}function _n(e){return gt(e)}class qc{constructor(t){this.auth=t,this.observer=null,this.addObserver=K0(n=>this.observer=n)}get next(){return O(this.observer,this.auth,"internal-error"),this.observer.next.bind(this.observer)}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */let qo={async loadJS(){throw new Error("Unable to load external scripts")},recaptchaV2Script:"",recaptchaEnterpriseScript:"",gapiScript:""};function Z1(e){qo=e}function Mp(e){return qo.loadJS(e)}function e_(){return qo.recaptchaEnterpriseScript}function t_(){return qo.gapiScript}function n_(e){return`__${e}${Math.floor(Math.random()*1e6)}`}const r_="recaptcha-enterprise",i_="NO_RECAPTCHA";class o_{constructor(t){this.type=r_,this.auth=_n(t)}async verify(t="verify",n=!1){async function r(o){if(!n){if(o.tenantId==null&&o._agentRecaptchaConfig!=null)return o._agentRecaptchaConfig.siteKey;if(o.tenantId!=null&&o._tenantRecaptchaConfigs[o.tenantId]!==void 0)return o._tenantRecaptchaConfigs[o.tenantId].siteKey}return new Promise(async(s,a)=>{U1(o,{clientType:"CLIENT_TYPE_WEB",version:"RECAPTCHA_ENTERPRISE"}).then(l=>{if(l.recaptchaKey===void 0)a(new Error("recaptcha Enterprise site key undefined"));else{const u=new M1(l);return o.tenantId==null?o._agentRecaptchaConfig=u:o._tenantRecaptchaConfigs[o.tenantId]=u,s(u.siteKey)}}).catch(l=>{a(l)})})}function i(o,s,a){const l=window.grecaptcha;Vc(l)?l.enterprise.ready(()=>{l.enterprise.execute(o,{action:t}).then(u=>{s(u)}).catch(()=>{s(i_)})}):a(Error("No reCAPTCHA enterprise script loaded."))}return new Promise((o,s)=>{r(this.auth).then(a=>{if(!n&&Vc(window.grecaptcha))i(a,o,s);else{if(typeof window>"u"){s(new Error("RecaptchaVerifier is only supported in browser"));return}let l=e_();l.length!==0&&(l+=a),Mp(l).then(()=>{i(a,o,s)}).catch(u=>{s(u)})}}).catch(a=>{s(a)})})}}async function Qc(e,t,n,r=!1){const i=new o_(e);let o;try{o=await i.verify(n)}catch{o=await i.verify(n,!0)}const s=Object.assign({},t);return r?Object.assign(s,{captchaResp:o}):Object.assign(s,{captchaResponse:o}),Object.assign(s,{clientType:"CLIENT_TYPE_WEB"}),Object.assign(s,{recaptchaVersion:"RECAPTCHA_ENTERPRISE"}),s}async function s_(e,t,n,r){var i;if(!((i=e._getRecaptchaConfig())===null||i===void 0)&&i.isProviderEnabled("EMAIL_PASSWORD_PROVIDER")){const o=await Qc(e,t,n,n==="getOobCode");return r(e,o)}else return r(e,t).catch(async o=>{if(o.code==="auth/missing-recaptcha-token"){console.log(`${n} is protected by reCAPTCHA Enterprise for this project. Automatically triggering the reCAPTCHA flow and restarting the flow.`);const s=await Qc(e,t,n,n==="getOobCode");return r(e,s)}else return Promise.reject(o)})}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function a_(e,t){const n=tr(e,"auth");if(n.isInitialized()){const i=n.getImmediate(),o=n.getOptions();if(Gr(o,t??{}))return i;Je(i,"already-initialized")}return n.initialize({options:t})}function l_(e,t){const n=(t==null?void 0:t.persistence)||[],r=(Array.isArray(n)?n:[n]).map(ot);t!=null&&t.errorMap&&e._updateErrorMap(t.errorMap),e._initializeWithPersistence(r,t==null?void 0:t.popupRedirectResolver)}function u_(e,t,n){const r=_n(e);O(r._canInitEmulator,r,"emulator-config-failed"),O(/^https?:\/\//.test(t),r,"invalid-emulator-scheme");const i=!1,o=Up(t),{host:s,port:a}=c_(t),l=a===null?"":`:${a}`;r.config.emulator={url:`${o}//${s}${l}/`},r.settings.appVerificationDisabledForTesting=!0,r.emulatorConfig=Object.freeze({host:s,port:a,protocol:o.replace(":",""),options:Object.freeze({disableWarnings:i})}),d_()}function Up(e){const t=e.indexOf(":");return t<0?"":e.substr(0,t+1)}function c_(e){const t=Up(e),n=/(\/\/)?([^?#/]+)/.exec(e.substr(t.length));if(!n)return{host:"",port:null};const r=n[2].split("@").pop()||"",i=/^(\[[^\]]+\])(:|$)/.exec(r);if(i){const o=i[1];return{host:o,port:Yc(r.substr(o.length+1))}}else{const[o,s]=r.split(":");return{host:o,port:Yc(s)}}}function Yc(e){if(!e)return null;const t=Number(e);return isNaN(t)?null:t}function d_(){function e(){const t=document.createElement("p"),n=t.style;t.innerText="Running in emulator mode. Do not use with production credentials.",n.position="fixed",n.width="100%",n.backgroundColor="#ffffff",n.border=".1em solid #000000",n.color="#b50000",n.bottom="0px",n.left="0px",n.margin="0px",n.zIndex="10000",n.textAlign="center",t.classList.add("firebase-emulator-warning"),document.body.appendChild(t)}typeof console<"u"&&typeof console.info=="function"&&console.info("WARNING: You are using the Auth Emulator, which is intended for local testing only.  Do not use with production credentials."),typeof window<"u"&&typeof document<"u"&&(document.readyState==="loading"?window.addEventListener("DOMContentLoaded",e):e())}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class jp{constructor(t,n){this.providerId=t,this.signInMethod=n}toJSON(){return rt("not implemented")}_getIdTokenResponse(t){return rt("not implemented")}_linkToIdToken(t,n){return rt("not implemented")}_getReauthenticationResolver(t){return rt("not implemented")}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Vn(e,t){return Ep(e,"POST","/v1/accounts:signInWithIdp",oi(e,t))}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const f_="http://localhost";class fn extends jp{constructor(){super(...arguments),this.pendingToken=null}static _fromParams(t){const n=new fn(t.providerId,t.signInMethod);return t.idToken||t.accessToken?(t.idToken&&(n.idToken=t.idToken),t.accessToken&&(n.accessToken=t.accessToken),t.nonce&&!t.pendingToken&&(n.nonce=t.nonce),t.pendingToken&&(n.pendingToken=t.pendingToken)):t.oauthToken&&t.oauthTokenSecret?(n.accessToken=t.oauthToken,n.secret=t.oauthTokenSecret):Je("argument-error"),n}toJSON(){return{idToken:this.idToken,accessToken:this.accessToken,secret:this.secret,nonce:this.nonce,pendingToken:this.pendingToken,providerId:this.providerId,signInMethod:this.signInMethod}}static fromJSON(t){const n=typeof t=="string"?JSON.parse(t):t,{providerId:r,signInMethod:i}=n,o=Yl(n,["providerId","signInMethod"]);if(!r||!i)return null;const s=new fn(r,i);return s.idToken=o.idToken||void 0,s.accessToken=o.accessToken||void 0,s.secret=o.secret,s.nonce=o.nonce,s.pendingToken=o.pendingToken||null,s}_getIdTokenResponse(t){const n=this.buildRequest();return Vn(t,n)}_linkToIdToken(t,n){const r=this.buildRequest();return r.idToken=n,Vn(t,r)}_getReauthenticationResolver(t){const n=this.buildRequest();return n.autoCreate=!1,Vn(t,n)}buildRequest(){const t={requestUri:f_,returnSecureToken:!0};if(this.pendingToken)t.pendingToken=this.pendingToken;else{const n={};this.idToken&&(n.id_token=this.idToken),this.accessToken&&(n.access_token=this.accessToken),this.secret&&(n.oauth_token_secret=this.secret),n.providerId=this.providerId,this.nonce&&!this.pendingToken&&(n.nonce=this.nonce),t.postBody=ni(n)}return t}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class nu{constructor(t){this.providerId=t,this.defaultLanguageCode=null,this.customParameters={}}setDefaultLanguage(t){this.defaultLanguageCode=t}setCustomParameters(t){return this.customParameters=t,this}getCustomParameters(){return this.customParameters}}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class si extends nu{constructor(){super(...arguments),this.scopes=[]}addScope(t){return this.scopes.includes(t)||this.scopes.push(t),this}getScopes(){return[...this.scopes]}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class It extends si{constructor(){super("facebook.com")}static credential(t){return fn._fromParams({providerId:It.PROVIDER_ID,signInMethod:It.FACEBOOK_SIGN_IN_METHOD,accessToken:t})}static credentialFromResult(t){return It.credentialFromTaggedObject(t)}static credentialFromError(t){return It.credentialFromTaggedObject(t.customData||{})}static credentialFromTaggedObject({_tokenResponse:t}){if(!t||!("oauthAccessToken"in t)||!t.oauthAccessToken)return null;try{return It.credential(t.oauthAccessToken)}catch{return null}}}It.FACEBOOK_SIGN_IN_METHOD="facebook.com";It.PROVIDER_ID="facebook.com";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class kt extends si{constructor(){super("google.com"),this.addScope("profile")}static credential(t,n){return fn._fromParams({providerId:kt.PROVIDER_ID,signInMethod:kt.GOOGLE_SIGN_IN_METHOD,idToken:t,accessToken:n})}static credentialFromResult(t){return kt.credentialFromTaggedObject(t)}static credentialFromError(t){return kt.credentialFromTaggedObject(t.customData||{})}static credentialFromTaggedObject({_tokenResponse:t}){if(!t)return null;const{oauthIdToken:n,oauthAccessToken:r}=t;if(!n&&!r)return null;try{return kt.credential(n,r)}catch{return null}}}kt.GOOGLE_SIGN_IN_METHOD="google.com";kt.PROVIDER_ID="google.com";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Ct extends si{constructor(){super("github.com")}static credential(t){return fn._fromParams({providerId:Ct.PROVIDER_ID,signInMethod:Ct.GITHUB_SIGN_IN_METHOD,accessToken:t})}static credentialFromResult(t){return Ct.credentialFromTaggedObject(t)}static credentialFromError(t){return Ct.credentialFromTaggedObject(t.customData||{})}static credentialFromTaggedObject({_tokenResponse:t}){if(!t||!("oauthAccessToken"in t)||!t.oauthAccessToken)return null;try{return Ct.credential(t.oauthAccessToken)}catch{return null}}}Ct.GITHUB_SIGN_IN_METHOD="github.com";Ct.PROVIDER_ID="github.com";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Tt extends si{constructor(){super("twitter.com")}static credential(t,n){return fn._fromParams({providerId:Tt.PROVIDER_ID,signInMethod:Tt.TWITTER_SIGN_IN_METHOD,oauthToken:t,oauthTokenSecret:n})}static credentialFromResult(t){return Tt.credentialFromTaggedObject(t)}static credentialFromError(t){return Tt.credentialFromTaggedObject(t.customData||{})}static credentialFromTaggedObject({_tokenResponse:t}){if(!t)return null;const{oauthAccessToken:n,oauthTokenSecret:r}=t;if(!n||!r)return null;try{return Tt.credential(n,r)}catch{return null}}}Tt.TWITTER_SIGN_IN_METHOD="twitter.com";Tt.PROVIDER_ID="twitter.com";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function h_(e,t){return Ep(e,"POST","/v1/accounts:signUp",oi(e,t))}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class hn{constructor(t){this.user=t.user,this.providerId=t.providerId,this._tokenResponse=t._tokenResponse,this.operationType=t.operationType}static async _fromIdTokenResponse(t,n,r,i=!1){const o=await it._fromIdTokenResponse(t,r,i),s=Jc(r);return new hn({user:o,providerId:s,_tokenResponse:r,operationType:n})}static async _forOperation(t,n,r){await t._updateTokensIfNecessary(r,!0);const i=Jc(r);return new hn({user:t,providerId:i,_tokenResponse:r,operationType:n})}}function Jc(e){return e.providerId?e.providerId:"phoneNumber"in e?"phone":null}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Eo extends Xe{constructor(t,n,r,i){var o;super(n.code,n.message),this.operationType=r,this.user=i,Object.setPrototypeOf(this,Eo.prototype),this.customData={appName:t.name,tenantId:(o=t.tenantId)!==null&&o!==void 0?o:void 0,_serverResponse:n.customData._serverResponse,operationType:r}}static _fromErrorAndOperation(t,n,r,i){return new Eo(t,n,r,i)}}function Fp(e,t,n,r){return(t==="reauthenticate"?n._getReauthenticationResolver(e):n._getIdTokenResponse(e)).catch(o=>{throw o.code==="auth/multi-factor-auth-required"?Eo._fromErrorAndOperation(e,o,t,r):o})}async function p_(e,t,n=!1){const r=await Qr(e,t._linkToIdToken(e.auth,await e.getIdToken()),n);return hn._forOperation(e,"link",r)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function m_(e,t,n=!1){const{auth:r}=e;if(He(r.app))return Promise.reject(lt(r));const i="reauthenticate";try{const o=await Qr(e,Fp(r,i,t,e),n);O(o.idToken,r,"internal-error");const s=eu(o.idToken);O(s,r,"internal-error");const{sub:a}=s;return O(e.uid===a,r,"user-mismatch"),hn._forOperation(e,i,o)}catch(o){throw(o==null?void 0:o.code)==="auth/user-not-found"&&Je(r,"user-mismatch"),o}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function g_(e,t,n=!1){if(He(e.app))return Promise.reject(lt(e));const r="signIn",i=await Fp(e,r,t),o=await hn._fromIdTokenResponse(e,r,i);return n||await e._updateCurrentUser(o.user),o}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function v_(e){const t=_n(e);t._getPasswordPolicyInternal()&&await t._updatePasswordPolicy()}async function y_(e,t,n){if(He(e.app))return Promise.reject(lt(e));const r=_n(e),s=await s_(r,{returnSecureToken:!0,email:t,password:n,clientType:"CLIENT_TYPE_WEB"},"signUpPassword",h_).catch(l=>{throw l.code==="auth/password-does-not-meet-requirements"&&v_(e),l}),a=await hn._fromIdTokenResponse(r,"signIn",s);return await r._updateCurrentUser(a.user),a}function w_(e,t,n,r){return gt(e).onIdTokenChanged(t,n,r)}function __(e,t,n){return gt(e).beforeAuthStateChanged(t,n)}function S_(e){return gt(e).signOut()}const Io="__sak";/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class zp{constructor(t,n){this.storageRetriever=t,this.type=n}_isAvailable(){try{return this.storage?(this.storage.setItem(Io,"1"),this.storage.removeItem(Io),Promise.resolve(!0)):Promise.resolve(!1)}catch{return Promise.resolve(!1)}}_set(t,n){return this.storage.setItem(t,JSON.stringify(n)),Promise.resolve()}_get(t){const n=this.storage.getItem(t);return Promise.resolve(n?JSON.parse(n):null)}_remove(t){return this.storage.removeItem(t),Promise.resolve()}get storage(){return this.storageRetriever()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function E_(){const e=ue();return tu(e)||Go(e)}const I_=1e3,k_=10;class bp extends zp{constructor(){super(()=>window.localStorage,"LOCAL"),this.boundEventHandler=(t,n)=>this.onStorageEvent(t,n),this.listeners={},this.localCache={},this.pollTimer=null,this.safariLocalStorageNotSynced=E_()&&G1(),this.fallbackToPolling=Lp(),this._shouldAllowMigration=!0}forAllChangedKeys(t){for(const n of Object.keys(this.listeners)){const r=this.storage.getItem(n),i=this.localCache[n];r!==i&&t(n,i,r)}}onStorageEvent(t,n=!1){if(!t.key){this.forAllChangedKeys((s,a,l)=>{this.notifyListeners(s,l)});return}const r=t.key;if(n?this.detachListener():this.stopPolling(),this.safariLocalStorageNotSynced){const s=this.storage.getItem(r);if(t.newValue!==s)t.newValue!==null?this.storage.setItem(r,t.newValue):this.storage.removeItem(r);else if(this.localCache[r]===t.newValue&&!n)return}const i=()=>{const s=this.storage.getItem(r);!n&&this.localCache[r]===s||this.notifyListeners(r,s)},o=this.storage.getItem(r);K1()&&o!==t.newValue&&t.newValue!==t.oldValue?setTimeout(i,k_):i()}notifyListeners(t,n){this.localCache[t]=n;const r=this.listeners[t];if(r)for(const i of Array.from(r))i(n&&JSON.parse(n))}startPolling(){this.stopPolling(),this.pollTimer=setInterval(()=>{this.forAllChangedKeys((t,n,r)=>{this.onStorageEvent(new StorageEvent("storage",{key:t,oldValue:n,newValue:r}),!0)})},I_)}stopPolling(){this.pollTimer&&(clearInterval(this.pollTimer),this.pollTimer=null)}attachListener(){window.addEventListener("storage",this.boundEventHandler)}detachListener(){window.removeEventListener("storage",this.boundEventHandler)}_addListener(t,n){Object.keys(this.listeners).length===0&&(this.fallbackToPolling?this.startPolling():this.attachListener()),this.listeners[t]||(this.listeners[t]=new Set,this.localCache[t]=this.storage.getItem(t)),this.listeners[t].add(n)}_removeListener(t,n){this.listeners[t]&&(this.listeners[t].delete(n),this.listeners[t].size===0&&delete this.listeners[t]),Object.keys(this.listeners).length===0&&(this.detachListener(),this.stopPolling())}async _set(t,n){await super._set(t,n),this.localCache[t]=JSON.stringify(n)}async _get(t){const n=await super._get(t);return this.localCache[t]=JSON.stringify(n),n}async _remove(t){await super._remove(t),delete this.localCache[t]}}bp.type="LOCAL";const C_=bp;/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class $p extends zp{constructor(){super(()=>window.sessionStorage,"SESSION")}_addListener(t,n){}_removeListener(t,n){}}$p.type="SESSION";const Bp=$p;/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function T_(e){return Promise.all(e.map(async t=>{try{return{fulfilled:!0,value:await t}}catch(n){return{fulfilled:!1,reason:n}}}))}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class Qo{constructor(t){this.eventTarget=t,this.handlersMap={},this.boundEventHandler=this.handleEvent.bind(this)}static _getInstance(t){const n=this.receivers.find(i=>i.isListeningto(t));if(n)return n;const r=new Qo(t);return this.receivers.push(r),r}isListeningto(t){return this.eventTarget===t}async handleEvent(t){const n=t,{eventId:r,eventType:i,data:o}=n.data,s=this.handlersMap[i];if(!(s!=null&&s.size))return;n.ports[0].postMessage({status:"ack",eventId:r,eventType:i});const a=Array.from(s).map(async u=>u(n.origin,o)),l=await T_(a);n.ports[0].postMessage({status:"done",eventId:r,eventType:i,response:l})}_subscribe(t,n){Object.keys(this.handlersMap).length===0&&this.eventTarget.addEventListener("message",this.boundEventHandler),this.handlersMap[t]||(this.handlersMap[t]=new Set),this.handlersMap[t].add(n)}_unsubscribe(t,n){this.handlersMap[t]&&n&&this.handlersMap[t].delete(n),(!n||this.handlersMap[t].size===0)&&delete this.handlersMap[t],Object.keys(this.handlersMap).length===0&&this.eventTarget.removeEventListener("message",this.boundEventHandler)}}Qo.receivers=[];/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function ru(e="",t=10){let n="";for(let r=0;r<t;r++)n+=Math.floor(Math.random()*10);return e+n}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class P_{constructor(t){this.target=t,this.handlers=new Set}removeMessageHandler(t){t.messageChannel&&(t.messageChannel.port1.removeEventListener("message",t.onMessage),t.messageChannel.port1.close()),this.handlers.delete(t)}async _send(t,n,r=50){const i=typeof MessageChannel<"u"?new MessageChannel:null;if(!i)throw new Error("connection_unavailable");let o,s;return new Promise((a,l)=>{const u=ru("",20);i.port1.start();const d=setTimeout(()=>{l(new Error("unsupported_event"))},r);s={messageChannel:i,onMessage(f){const m=f;if(m.data.eventId===u)switch(m.data.status){case"ack":clearTimeout(d),o=setTimeout(()=>{l(new Error("timeout"))},3e3);break;case"done":clearTimeout(o),a(m.data.response);break;default:clearTimeout(d),clearTimeout(o),l(new Error("invalid_response"));break}}},this.handlers.add(s),i.port1.addEventListener("message",s.onMessage),this.target.postMessage({eventType:t,eventId:u,data:n},[i.port2])}).finally(()=>{s&&this.removeMessageHandler(s)})}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Qe(){return window}function R_(e){Qe().location.href=e}/**
 * @license
 * Copyright 2020 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Vp(){return typeof Qe().WorkerGlobalScope<"u"&&typeof Qe().importScripts=="function"}async function N_(){if(!(navigator!=null&&navigator.serviceWorker))return null;try{return(await navigator.serviceWorker.ready).active}catch{return null}}function O_(){var e;return((e=navigator==null?void 0:navigator.serviceWorker)===null||e===void 0?void 0:e.controller)||null}function A_(){return Vp()?self:null}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Hp="firebaseLocalStorageDb",x_=1,ko="firebaseLocalStorage",Wp="fbase_key";class ai{constructor(t){this.request=t}toPromise(){return new Promise((t,n)=>{this.request.addEventListener("success",()=>{t(this.request.result)}),this.request.addEventListener("error",()=>{n(this.request.error)})})}}function Yo(e,t){return e.transaction([ko],t?"readwrite":"readonly").objectStore(ko)}function L_(){const e=indexedDB.deleteDatabase(Hp);return new ai(e).toPromise()}function Va(){const e=indexedDB.open(Hp,x_);return new Promise((t,n)=>{e.addEventListener("error",()=>{n(e.error)}),e.addEventListener("upgradeneeded",()=>{const r=e.result;try{r.createObjectStore(ko,{keyPath:Wp})}catch(i){n(i)}}),e.addEventListener("success",async()=>{const r=e.result;r.objectStoreNames.contains(ko)?t(r):(r.close(),await L_(),t(await Va()))})})}async function Xc(e,t,n){const r=Yo(e,!0).put({[Wp]:t,value:n});return new ai(r).toPromise()}async function D_(e,t){const n=Yo(e,!1).get(t),r=await new ai(n).toPromise();return r===void 0?null:r.value}function Zc(e,t){const n=Yo(e,!0).delete(t);return new ai(n).toPromise()}const M_=800,U_=3;class Kp{constructor(){this.type="LOCAL",this._shouldAllowMigration=!0,this.listeners={},this.localCache={},this.pollTimer=null,this.pendingWrites=0,this.receiver=null,this.sender=null,this.serviceWorkerReceiverAvailable=!1,this.activeServiceWorker=null,this._workerInitializationPromise=this.initializeServiceWorkerMessaging().then(()=>{},()=>{})}async _openDb(){return this.db?this.db:(this.db=await Va(),this.db)}async _withRetries(t){let n=0;for(;;)try{const r=await this._openDb();return await t(r)}catch(r){if(n++>U_)throw r;this.db&&(this.db.close(),this.db=void 0)}}async initializeServiceWorkerMessaging(){return Vp()?this.initializeReceiver():this.initializeSender()}async initializeReceiver(){this.receiver=Qo._getInstance(A_()),this.receiver._subscribe("keyChanged",async(t,n)=>({keyProcessed:(await this._poll()).includes(n.key)})),this.receiver._subscribe("ping",async(t,n)=>["keyChanged"])}async initializeSender(){var t,n;if(this.activeServiceWorker=await N_(),!this.activeServiceWorker)return;this.sender=new P_(this.activeServiceWorker);const r=await this.sender._send("ping",{},800);r&&!((t=r[0])===null||t===void 0)&&t.fulfilled&&!((n=r[0])===null||n===void 0)&&n.value.includes("keyChanged")&&(this.serviceWorkerReceiverAvailable=!0)}async notifyServiceWorker(t){if(!(!this.sender||!this.activeServiceWorker||O_()!==this.activeServiceWorker))try{await this.sender._send("keyChanged",{key:t},this.serviceWorkerReceiverAvailable?800:50)}catch{}}async _isAvailable(){try{if(!indexedDB)return!1;const t=await Va();return await Xc(t,Io,"1"),await Zc(t,Io),!0}catch{}return!1}async _withPendingWrite(t){this.pendingWrites++;try{await t()}finally{this.pendingWrites--}}async _set(t,n){return this._withPendingWrite(async()=>(await this._withRetries(r=>Xc(r,t,n)),this.localCache[t]=n,this.notifyServiceWorker(t)))}async _get(t){const n=await this._withRetries(r=>D_(r,t));return this.localCache[t]=n,n}async _remove(t){return this._withPendingWrite(async()=>(await this._withRetries(n=>Zc(n,t)),delete this.localCache[t],this.notifyServiceWorker(t)))}async _poll(){const t=await this._withRetries(i=>{const o=Yo(i,!1).getAll();return new ai(o).toPromise()});if(!t)return[];if(this.pendingWrites!==0)return[];const n=[],r=new Set;if(t.length!==0)for(const{fbase_key:i,value:o}of t)r.add(i),JSON.stringify(this.localCache[i])!==JSON.stringify(o)&&(this.notifyListeners(i,o),n.push(i));for(const i of Object.keys(this.localCache))this.localCache[i]&&!r.has(i)&&(this.notifyListeners(i,null),n.push(i));return n}notifyListeners(t,n){this.localCache[t]=n;const r=this.listeners[t];if(r)for(const i of Array.from(r))i(n)}startPolling(){this.stopPolling(),this.pollTimer=setInterval(async()=>this._poll(),M_)}stopPolling(){this.pollTimer&&(clearInterval(this.pollTimer),this.pollTimer=null)}_addListener(t,n){Object.keys(this.listeners).length===0&&this.startPolling(),this.listeners[t]||(this.listeners[t]=new Set,this._get(t)),this.listeners[t].add(n)}_removeListener(t,n){this.listeners[t]&&(this.listeners[t].delete(n),this.listeners[t].size===0&&delete this.listeners[t]),Object.keys(this.listeners).length===0&&this.stopPolling()}}Kp.type="LOCAL";const j_=Kp;new ii(3e4,6e4);/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function Gp(e,t){return t?ot(t):(O(e._popupRedirectResolver,e,"argument-error"),e._popupRedirectResolver)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class iu extends jp{constructor(t){super("custom","custom"),this.params=t}_getIdTokenResponse(t){return Vn(t,this._buildIdpRequest())}_linkToIdToken(t,n){return Vn(t,this._buildIdpRequest(n))}_getReauthenticationResolver(t){return Vn(t,this._buildIdpRequest())}_buildIdpRequest(t){const n={requestUri:this.params.requestUri,sessionId:this.params.sessionId,postBody:this.params.postBody,tenantId:this.params.tenantId,pendingToken:this.params.pendingToken,returnSecureToken:!0,returnIdpCredential:!0};return t&&(n.idToken=t),n}}function F_(e){return g_(e.auth,new iu(e),e.bypassAuthState)}function z_(e){const{auth:t,user:n}=e;return O(n,t,"internal-error"),m_(n,new iu(e),e.bypassAuthState)}async function b_(e){const{auth:t,user:n}=e;return O(n,t,"internal-error"),p_(n,new iu(e),e.bypassAuthState)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class qp{constructor(t,n,r,i,o=!1){this.auth=t,this.resolver=r,this.user=i,this.bypassAuthState=o,this.pendingPromise=null,this.eventManager=null,this.filter=Array.isArray(n)?n:[n]}execute(){return new Promise(async(t,n)=>{this.pendingPromise={resolve:t,reject:n};try{this.eventManager=await this.resolver._initialize(this.auth),await this.onExecution(),this.eventManager.registerConsumer(this)}catch(r){this.reject(r)}})}async onAuthEvent(t){const{urlResponse:n,sessionId:r,postBody:i,tenantId:o,error:s,type:a}=t;if(s){this.reject(s);return}const l={auth:this.auth,requestUri:n,sessionId:r,tenantId:o||void 0,postBody:i||void 0,user:this.user,bypassAuthState:this.bypassAuthState};try{this.resolve(await this.getIdpTask(a)(l))}catch(u){this.reject(u)}}onError(t){this.reject(t)}getIdpTask(t){switch(t){case"signInViaPopup":case"signInViaRedirect":return F_;case"linkViaPopup":case"linkViaRedirect":return b_;case"reauthViaPopup":case"reauthViaRedirect":return z_;default:Je(this.auth,"internal-error")}}resolve(t){pt(this.pendingPromise,"Pending promise was never set"),this.pendingPromise.resolve(t),this.unregisterAndCleanUp()}reject(t){pt(this.pendingPromise,"Pending promise was never set"),this.pendingPromise.reject(t),this.unregisterAndCleanUp()}unregisterAndCleanUp(){this.eventManager&&this.eventManager.unregisterConsumer(this),this.pendingPromise=null,this.cleanUp()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const $_=new ii(2e3,1e4);class Dn extends qp{constructor(t,n,r,i,o){super(t,n,i,o),this.provider=r,this.authWindow=null,this.pollId=null,Dn.currentPopupAction&&Dn.currentPopupAction.cancel(),Dn.currentPopupAction=this}async executeNotNull(){const t=await this.execute();return O(t,this.auth,"internal-error"),t}async onExecution(){pt(this.filter.length===1,"Popup operations only handle one event");const t=ru();this.authWindow=await this.resolver._openPopup(this.auth,this.provider,this.filter[0],t),this.authWindow.associatedEvent=t,this.resolver._originValidation(this.auth).catch(n=>{this.reject(n)}),this.resolver._isIframeWebStorageSupported(this.auth,n=>{n||this.reject(qe(this.auth,"web-storage-unsupported"))}),this.pollUserCancellation()}get eventId(){var t;return((t=this.authWindow)===null||t===void 0?void 0:t.associatedEvent)||null}cancel(){this.reject(qe(this.auth,"cancelled-popup-request"))}cleanUp(){this.authWindow&&this.authWindow.close(),this.pollId&&window.clearTimeout(this.pollId),this.authWindow=null,this.pollId=null,Dn.currentPopupAction=null}pollUserCancellation(){const t=()=>{var n,r;if(!((r=(n=this.authWindow)===null||n===void 0?void 0:n.window)===null||r===void 0)&&r.closed){this.pollId=window.setTimeout(()=>{this.pollId=null,this.reject(qe(this.auth,"popup-closed-by-user"))},8e3);return}this.pollId=window.setTimeout(t,$_.get())};t()}}Dn.currentPopupAction=null;/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const B_="pendingRedirect",Bi=new Map;class V_ extends qp{constructor(t,n,r=!1){super(t,["signInViaRedirect","linkViaRedirect","reauthViaRedirect","unknown"],n,void 0,r),this.eventId=null}async execute(){let t=Bi.get(this.auth._key());if(!t){try{const r=await H_(this.resolver,this.auth)?await super.execute():null;t=()=>Promise.resolve(r)}catch(n){t=()=>Promise.reject(n)}Bi.set(this.auth._key(),t)}return this.bypassAuthState||Bi.set(this.auth._key(),()=>Promise.resolve(null)),t()}async onAuthEvent(t){if(t.type==="signInViaRedirect")return super.onAuthEvent(t);if(t.type==="unknown"){this.resolve(null);return}if(t.eventId){const n=await this.auth._redirectUserForId(t.eventId);if(n)return this.user=n,super.onAuthEvent(t);this.resolve(null)}}async onExecution(){}cleanUp(){}}async function H_(e,t){const n=Yp(t),r=Qp(e);if(!await r._isAvailable())return!1;const i=await r._get(n)==="true";return await r._remove(n),i}async function W_(e,t){return Qp(e)._set(Yp(t),"true")}function K_(e,t){Bi.set(e._key(),t)}function Qp(e){return ot(e._redirectPersistence)}function Yp(e){return $i(B_,e.config.apiKey,e.name)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function G_(e,t,n){return q_(e,t,n)}async function q_(e,t,n){if(He(e.app))return Promise.reject(lt(e));const r=_n(e);P1(e,t,nu),await r._initializationPromise;const i=Gp(r,n);return await W_(i,r),i._openRedirect(r,t,"signInViaRedirect")}async function Q_(e,t,n=!1){if(He(e.app))return Promise.reject(lt(e));const r=_n(e),i=Gp(r,t),s=await new V_(r,i,n).execute();return s&&!n&&(delete s.user._redirectEventId,await r._persistUserIfCurrent(s.user),await r._setRedirectUser(null,t)),s}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const Y_=10*60*1e3;class J_{constructor(t){this.auth=t,this.cachedEventUids=new Set,this.consumers=new Set,this.queuedRedirectEvent=null,this.hasHandledPotentialRedirect=!1,this.lastProcessedEventTime=Date.now()}registerConsumer(t){this.consumers.add(t),this.queuedRedirectEvent&&this.isEventForConsumer(this.queuedRedirectEvent,t)&&(this.sendToConsumer(this.queuedRedirectEvent,t),this.saveEventToCache(this.queuedRedirectEvent),this.queuedRedirectEvent=null)}unregisterConsumer(t){this.consumers.delete(t)}onEvent(t){if(this.hasEventBeenHandled(t))return!1;let n=!1;return this.consumers.forEach(r=>{this.isEventForConsumer(t,r)&&(n=!0,this.sendToConsumer(t,r),this.saveEventToCache(t))}),this.hasHandledPotentialRedirect||!X_(t)||(this.hasHandledPotentialRedirect=!0,n||(this.queuedRedirectEvent=t,n=!0)),n}sendToConsumer(t,n){var r;if(t.error&&!Jp(t)){const i=((r=t.error.code)===null||r===void 0?void 0:r.split("auth/")[1])||"internal-error";n.onError(qe(this.auth,i))}else n.onAuthEvent(t)}isEventForConsumer(t,n){const r=n.eventId===null||!!t.eventId&&t.eventId===n.eventId;return n.filter.includes(t.type)&&r}hasEventBeenHandled(t){return Date.now()-this.lastProcessedEventTime>=Y_&&this.cachedEventUids.clear(),this.cachedEventUids.has(ed(t))}saveEventToCache(t){this.cachedEventUids.add(ed(t)),this.lastProcessedEventTime=Date.now()}}function ed(e){return[e.type,e.eventId,e.sessionId,e.tenantId].filter(t=>t).join("-")}function Jp({type:e,error:t}){return e==="unknown"&&(t==null?void 0:t.code)==="auth/no-auth-event"}function X_(e){switch(e.type){case"signInViaRedirect":case"linkViaRedirect":case"reauthViaRedirect":return!0;case"unknown":return Jp(e);default:return!1}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */async function Z_(e,t={}){return wn(e,"GET","/v1/projects",t)}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const eS=/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/,tS=/^https?/;async function nS(e){if(e.config.emulator)return;const{authorizedDomains:t}=await Z_(e);for(const n of t)try{if(rS(n))return}catch{}Je(e,"unauthorized-domain")}function rS(e){const t=$a(),{protocol:n,hostname:r}=new URL(t);if(e.startsWith("chrome-extension://")){const s=new URL(e);return s.hostname===""&&r===""?n==="chrome-extension:"&&e.replace("chrome-extension://","")===t.replace("chrome-extension://",""):n==="chrome-extension:"&&s.hostname===r}if(!tS.test(n))return!1;if(eS.test(e))return r===e;const i=e.replace(/\./g,"\\.");return new RegExp("^(.+\\."+i+"|"+i+")$","i").test(r)}/**
 * @license
 * Copyright 2020 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const iS=new ii(3e4,6e4);function td(){const e=Qe().___jsl;if(e!=null&&e.H){for(const t of Object.keys(e.H))if(e.H[t].r=e.H[t].r||[],e.H[t].L=e.H[t].L||[],e.H[t].r=[...e.H[t].L],e.CP)for(let n=0;n<e.CP.length;n++)e.CP[n]=null}}function oS(e){return new Promise((t,n)=>{var r,i,o;function s(){td(),gapi.load("gapi.iframes",{callback:()=>{t(gapi.iframes.getContext())},ontimeout:()=>{td(),n(qe(e,"network-request-failed"))},timeout:iS.get()})}if(!((i=(r=Qe().gapi)===null||r===void 0?void 0:r.iframes)===null||i===void 0)&&i.Iframe)t(gapi.iframes.getContext());else if(!((o=Qe().gapi)===null||o===void 0)&&o.load)s();else{const a=n_("iframefcb");return Qe()[a]=()=>{gapi.load?s():n(qe(e,"network-request-failed"))},Mp(`${t_()}?onload=${a}`).catch(l=>n(l))}}).catch(t=>{throw Vi=null,t})}let Vi=null;function sS(e){return Vi=Vi||oS(e),Vi}/**
 * @license
 * Copyright 2020 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const aS=new ii(5e3,15e3),lS="__/auth/iframe",uS="emulator/auth/iframe",cS={style:{position:"absolute",top:"-100px",width:"1px",height:"1px"},"aria-hidden":"true",tabindex:"-1"},dS=new Map([["identitytoolkit.googleapis.com","p"],["staging-identitytoolkit.sandbox.googleapis.com","s"],["test-identitytoolkit.sandbox.googleapis.com","t"]]);function fS(e){const t=e.config;O(t.authDomain,e,"auth-domain-config-required");const n=t.emulator?Zl(t,uS):`https://${e.config.authDomain}/${lS}`,r={apiKey:t.apiKey,appName:e.name,v:ri},i=dS.get(e.config.apiHost);i&&(r.eid=i);const o=e._getFrameworks();return o.length&&(r.fw=o.join(",")),`${n}?${ni(r).slice(1)}`}async function hS(e){const t=await sS(e),n=Qe().gapi;return O(n,e,"internal-error"),t.open({where:document.body,url:fS(e),messageHandlersFilter:n.iframes.CROSS_ORIGIN_IFRAMES_FILTER,attributes:cS,dontclear:!0},r=>new Promise(async(i,o)=>{await r.restyle({setHideOnLeave:!1});const s=qe(e,"network-request-failed"),a=Qe().setTimeout(()=>{o(s)},aS.get());function l(){Qe().clearTimeout(a),i(r)}r.ping(l).then(l,()=>{o(s)})}))}/**
 * @license
 * Copyright 2020 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const pS={location:"yes",resizable:"yes",statusbar:"yes",toolbar:"no"},mS=500,gS=600,vS="_blank",yS="http://localhost";class nd{constructor(t){this.window=t,this.associatedEvent=null}close(){if(this.window)try{this.window.close()}catch{}}}function wS(e,t,n,r=mS,i=gS){const o=Math.max((window.screen.availHeight-i)/2,0).toString(),s=Math.max((window.screen.availWidth-r)/2,0).toString();let a="";const l=Object.assign(Object.assign({},pS),{width:r.toString(),height:i.toString(),top:o,left:s}),u=ue().toLowerCase();n&&(a=Rp(u)?vS:n),Pp(u)&&(t=t||yS,l.scrollbars="yes");const d=Object.entries(l).reduce((m,[g,y])=>`${m}${g}=${y},`,"");if(W1(u)&&a!=="_self")return _S(t||"",a),new nd(null);const f=window.open(t||"",a,d);O(f,e,"popup-blocked");try{f.focus()}catch{}return new nd(f)}function _S(e,t){const n=document.createElement("a");n.href=e,n.target=t;const r=document.createEvent("MouseEvent");r.initMouseEvent("click",!0,!0,window,1,0,0,0,0,!1,!1,!1,!1,1,null),n.dispatchEvent(r)}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const SS="__/auth/handler",ES="emulator/auth/handler",IS=encodeURIComponent("fac");async function rd(e,t,n,r,i,o){O(e.config.authDomain,e,"auth-domain-config-required"),O(e.config.apiKey,e,"invalid-api-key");const s={apiKey:e.config.apiKey,appName:e.name,authType:n,redirectUrl:r,v:ri,eventId:i};if(t instanceof nu){t.setDefaultLanguage(e.languageCode),s.providerId=t.providerId||"",W0(t.getCustomParameters())||(s.customParameters=JSON.stringify(t.getCustomParameters()));for(const[d,f]of Object.entries({}))s[d]=f}if(t instanceof si){const d=t.getScopes().filter(f=>f!=="");d.length>0&&(s.scopes=d.join(","))}e.tenantId&&(s.tid=e.tenantId);const a=s;for(const d of Object.keys(a))a[d]===void 0&&delete a[d];const l=await e._getAppCheckToken(),u=l?`#${IS}=${encodeURIComponent(l)}`:"";return`${kS(e)}?${ni(a).slice(1)}${u}`}function kS({config:e}){return e.emulator?Zl(e,ES):`https://${e.authDomain}/${SS}`}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const js="webStorageSupport";class CS{constructor(){this.eventManagers={},this.iframes={},this.originValidationPromises={},this._redirectPersistence=Bp,this._completeRedirectFn=Q_,this._overrideRedirectResult=K_}async _openPopup(t,n,r,i){var o;pt((o=this.eventManagers[t._key()])===null||o===void 0?void 0:o.manager,"_initialize() not called before _openPopup()");const s=await rd(t,n,r,$a(),i);return wS(t,s,ru())}async _openRedirect(t,n,r,i){await this._originValidation(t);const o=await rd(t,n,r,$a(),i);return R_(o),new Promise(()=>{})}_initialize(t){const n=t._key();if(this.eventManagers[n]){const{manager:i,promise:o}=this.eventManagers[n];return i?Promise.resolve(i):(pt(o,"If manager is not set, promise should be"),o)}const r=this.initAndGetManager(t);return this.eventManagers[n]={promise:r},r.catch(()=>{delete this.eventManagers[n]}),r}async initAndGetManager(t){const n=await hS(t),r=new J_(t);return n.register("authEvent",i=>(O(i==null?void 0:i.authEvent,t,"invalid-auth-event"),{status:r.onEvent(i.authEvent)?"ACK":"ERROR"}),gapi.iframes.CROSS_ORIGIN_IFRAMES_FILTER),this.eventManagers[t._key()]={manager:r},this.iframes[t._key()]=n,r}_isIframeWebStorageSupported(t,n){this.iframes[t._key()].send(js,{type:js},i=>{var o;const s=(o=i==null?void 0:i[0])===null||o===void 0?void 0:o[js];s!==void 0&&n(!!s),Je(t,"internal-error")},gapi.iframes.CROSS_ORIGIN_IFRAMES_FILTER)}_originValidation(t){const n=t._key();return this.originValidationPromises[n]||(this.originValidationPromises[n]=nS(t)),this.originValidationPromises[n]}get _shouldInitProactively(){return Lp()||tu()||Go()}}const TS=CS;var id="@firebase/auth",od="1.7.4";/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */class PS{constructor(t){this.auth=t,this.internalListeners=new Map}getUid(){var t;return this.assertAuthConfigured(),((t=this.auth.currentUser)===null||t===void 0?void 0:t.uid)||null}async getToken(t){return this.assertAuthConfigured(),await this.auth._initializationPromise,this.auth.currentUser?{accessToken:await this.auth.currentUser.getIdToken(t)}:null}addAuthTokenListener(t){if(this.assertAuthConfigured(),this.internalListeners.has(t))return;const n=this.auth.onIdTokenChanged(r=>{t((r==null?void 0:r.stsTokenManager.accessToken)||null)});this.internalListeners.set(t,n),this.updateProactiveRefresh()}removeAuthTokenListener(t){this.assertAuthConfigured();const n=this.internalListeners.get(t);n&&(this.internalListeners.delete(t),n(),this.updateProactiveRefresh())}assertAuthConfigured(){O(this.auth._initializationPromise,"dependent-sdk-initialized-before-auth")}updateProactiveRefresh(){this.internalListeners.size>0?this.auth._startProactiveRefresh():this.auth._stopProactiveRefresh()}}/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */function RS(e){switch(e){case"Node":return"node";case"ReactNative":return"rn";case"Worker":return"webworker";case"Cordova":return"cordova";case"WebExtension":return"web-extension";default:return}}function NS(e){ht(new Ye("auth",(t,{options:n})=>{const r=t.getProvider("app").getImmediate(),i=t.getProvider("heartbeat"),o=t.getProvider("app-check-internal"),{apiKey:s,authDomain:a}=r.options;O(s&&!s.includes(":"),"invalid-api-key",{appName:r.name});const l={apiKey:s,authDomain:a,clientPlatform:e,apiHost:"identitytoolkit.googleapis.com",tokenApiHost:"securetoken.googleapis.com",apiScheme:"https",sdkClientVersion:Dp(e)},u=new X1(r,i,o,l);return l_(u,n),u},"PUBLIC").setInstantiationMode("EXPLICIT").setInstanceCreatedCallback((t,n,r)=>{t.getProvider("auth-internal").initialize()})),ht(new Ye("auth-internal",t=>{const n=_n(t.getProvider("auth").getImmediate());return(r=>new PS(r))(n)},"PRIVATE").setInstantiationMode("EXPLICIT")),Ge(id,od,RS(e)),Ge(id,od,"esm2017")}/**
 * @license
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */const OS=5*60,AS=zh("authIdTokenMaxAge")||OS;let sd=null;const xS=e=>async t=>{const n=t&&await t.getIdTokenResult(),r=n&&(new Date().getTime()-Date.parse(n.issuedAtTime))/1e3;if(r&&r>AS)return;const i=n==null?void 0:n.token;sd!==i&&(sd=i,await fetch(e,{method:i?"POST":"DELETE",headers:i?{Authorization:`Bearer ${i}`}:{}}))};function LS(e=Gh()){const t=tr(e,"auth");if(t.isInitialized())return t.getImmediate();const n=a_(e,{popupRedirectResolver:TS,persistence:[j_,C_,Bp]}),r=zh("authTokenSyncURL");if(r&&typeof isSecureContext=="boolean"&&isSecureContext){const o=new URL(r,location.origin);if(location.origin===o.origin){const s=xS(o.toString());__(n,s,()=>s(n.currentUser)),w_(n,a=>s(a))}}const i=U0("auth");return i&&u_(n,`http://${i}`),n}function DS(){var e,t;return(t=(e=document.getElementsByTagName("head"))===null||e===void 0?void 0:e[0])!==null&&t!==void 0?t:document}Z1({loadJS(e){return new Promise((t,n)=>{const r=document.createElement("script");r.setAttribute("src",e),r.onload=t,r.onerror=i=>{const o=qe("internal-error");o.customData=i,n(o)},r.type="text/javascript",r.charset="UTF-8",DS().appendChild(r)})},gapiScript:"https://apis.google.com/js/api.js",recaptchaV2Script:"https://www.google.com/recaptcha/api.js",recaptchaEnterpriseScript:"https://www.google.com/recaptcha/enterprise.js?render="});NS("Browser");const MS={apiKey:"AIzaSyCekod31iBbxlYkSdf4G4PgkmWwGiQ0U-Y",authDomain:"fakenewsdetection-website.firebaseapp.com",projectId:"fakenewsdetection-website",storageBucket:"fakenewsdetection-website.appspot.com",messagingSenderId:"886470479542",appId:"1:886470479542:web:837d4fc93da491639ba017",measurementId:"G-LF6FQF3HED"},Xp=Kh(MS);S1(Xp);const ou=LS(Xp),US=()=>{const[e,t]=I.useState(""),[n,r]=I.useState(""),i=async o=>{o.preventDefault();try{await G_(ou,e,n),console.log("login successful"),move("/Home.jsx")}catch(s){alert(s.code)}};return k.jsxs("div",{className:"signCard",style:{backgroundImage:"url("+Eh+")"},children:[k.jsx("img",{id:"logo",src:Fl,alt:"logo"}),k.jsxs("form",{id:"signin-form",onSubmit:i,children:[k.jsx("p",{className:"Welcome",children:"Welcome back! "}),k.jsxs("fieldset",{children:[k.jsxs("label",{htmlFor:"email-address",children:["Email Address:",k.jsx("input",{type:"email",id:"email-address",required:!0,value:e,onChange:o=>t(o.target.value)})]}),k.jsxs("label",{htmlFor:"passKey",children:["Password:",k.jsx("input",{type:"password",id:"passKey",required:!0,value:n,onChange:o=>r(o.target.value)}),k.jsx("p",{children:k.jsx("a",{href:"",children:"Forgot password?"})})]})]}),k.jsxs("fieldset",{children:[k.jsx("button",{type:"submit",className:"submit-button",children:"Sign In"}),k.jsxs("p",{children:["Don't have an account?",k.jsx(Da,{className:"signUp",to:"/SignUp.jsx",children:"SignUp"})]})]})]})]})},jS=()=>{const[e,t]=I.useState(""),[n,r]=I.useState(""),[i,o]=I.useState(""),s=$l(),a=async l=>{l.preventDefault();try{await y_(ou,n,i),console.log("account created"),s("/Home.jsx")}catch(u){console.log(u)}};return k.jsxs("div",{className:"signCard",style:{backgroundImage:"url("+Eh+")"},children:[k.jsx("img",{id:"logo",src:Fl,alt:"logo"}),k.jsxs("form",{id:"signup-form",onSubmit:a,children:[k.jsx("p",{className:"Welcome",children:"Get Started "}),k.jsxs("fieldset",{children:[k.jsxs("label",{htmlFor:"username",children:["Username:",k.jsx("input",{type:"text",id:"username",required:!0,value:e,onChange:l=>t(l.target.value)})]}),k.jsxs("label",{htmlFor:"email-address",children:["Email Address:",k.jsx("input",{type:"email",id:"email-address",required:!0,value:n,onChange:l=>r(l.target.value)})]}),k.jsxs("label",{htmlFor:"passKey",children:["Password:",k.jsx("input",{type:"password",id:"passKey",required:!0,value:i,onChange:l=>o(l.target.value)})]})]}),k.jsx("input",{className:"terms",type:"radio",name:"terms-conditions",value:"true"}),k.jsx("span",{children:"I agree to to the"}),k.jsx(Da,{className:"terms-conditions",to:"/TermsAndConditions.jsx",children:"Terms and Conditions"}),k.jsxs("fieldset",{children:[k.jsx("button",{type:"submit",className:"submit-button",children:"Sign Up"}),k.jsxs("p",{children:["Have an account already?",k.jsx(Da,{className:"signIn",to:"/",children:"SignIn"})]})]})]})]})},FS=()=>k.jsx("div",{className:"Terms",children:k.jsx("h1",{children:"Terms and Conditions"})});var Zp={color:void 0,size:void 0,className:void 0,style:void 0,attr:void 0},ad=st.createContext&&st.createContext(Zp),zS=["attr","size","title"];function bS(e,t){if(e==null)return{};var n=$S(e,t),r,i;if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(i=0;i<o.length;i++)r=o[i],!(t.indexOf(r)>=0)&&Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}function $S(e,t){if(e==null)return{};var n={};for(var r in e)if(Object.prototype.hasOwnProperty.call(e,r)){if(t.indexOf(r)>=0)continue;n[r]=e[r]}return n}function Co(){return Co=Object.assign?Object.assign.bind():function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},Co.apply(this,arguments)}function ld(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter(function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable})),n.push.apply(n,r)}return n}function To(e){for(var t=1;t<arguments.length;t++){var n=arguments[t]!=null?arguments[t]:{};t%2?ld(Object(n),!0).forEach(function(r){BS(e,r,n[r])}):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):ld(Object(n)).forEach(function(r){Object.defineProperty(e,r,Object.getOwnPropertyDescriptor(n,r))})}return e}function BS(e,t,n){return t=VS(t),t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function VS(e){var t=HS(e,"string");return typeof t=="symbol"?t:t+""}function HS(e,t){if(typeof e!="object"||!e)return e;var n=e[Symbol.toPrimitive];if(n!==void 0){var r=n.call(e,t||"default");if(typeof r!="object")return r;throw new TypeError("@@toPrimitive must return a primitive value.")}return(t==="string"?String:Number)(e)}function em(e){return e&&e.map((t,n)=>st.createElement(t.tag,To({key:n},t.attr),em(t.child)))}function Jo(e){return t=>st.createElement(WS,Co({attr:To({},e.attr)},t),em(e.child))}function WS(e){var t=n=>{var{attr:r,size:i,title:o}=e,s=bS(e,zS),a=i||n.size||"1em",l;return n.className&&(l=n.className),e.className&&(l=(l?l+" ":"")+e.className),st.createElement("svg",Co({stroke:"currentColor",fill:"currentColor",strokeWidth:"0"},n.attr,r,s,{className:l,style:To(To({color:e.color||n.color},n.style),e.style),height:a,width:a,xmlns:"http://www.w3.org/2000/svg"}),o&&st.createElement("title",null,o),e.children)};return ad!==void 0?st.createElement(ad.Consumer,null,n=>t(n)):t(Zp)}function KS(e){return Jo({tag:"svg",attr:{viewBox:"0 0 24 24",fill:"currentColor"},child:[{tag:"path",attr:{d:"M14 3V5H4V18.3851L5.76282 17H20V10H22V18C22 18.5523 21.5523 19 21 19H6.45455L2 22.5V4C2 3.44772 2.44772 3 3 3H14ZM19 3V0H21V3H24V5H21V8H19V5H16V3H19Z"},child:[]}]})(e)}function GS(e){return Jo({tag:"svg",attr:{viewBox:"0 0 24 24",fill:"currentColor"},child:[{tag:"path",attr:{d:"M6.45455 19L2 22.5V4C2 3.44772 2.44772 3 3 3H21C21.5523 3 22 3.44772 22 4V18C22 18.5523 21.5523 19 21 19H6.45455ZM4 18.3851L5.76282 17H20V5H4V18.3851ZM13 11V15H11V11H8L12 7L16 11H13Z"},child:[]}]})(e)}function qS(e){return Jo({tag:"svg",attr:{viewBox:"0 0 24 24",fill:"currentColor"},child:[{tag:"path",attr:{d:"M12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12H4C4 16.4183 7.58172 20 12 20C16.4183 20 20 16.4183 20 12C20 7.58172 16.4183 4 12 4C9.25022 4 6.82447 5.38734 5.38451 7.50024L8 7.5V9.5H2V3.5H4L3.99989 5.99918C5.82434 3.57075 8.72873 2 12 2ZM13 7L12.9998 11.585L16.2426 14.8284L14.8284 16.2426L10.9998 12.413L11 7H13Z"},child:[]}]})(e)}function QS(e){return Jo({tag:"svg",attr:{viewBox:"0 0 24 24",fill:"currentColor"},child:[{tag:"path",attr:{d:"M8.68637 4.00008L11.293 1.39348C11.6835 1.00295 12.3167 1.00295 12.7072 1.39348L15.3138 4.00008H19.0001C19.5524 4.00008 20.0001 4.4478 20.0001 5.00008V8.68637L22.6067 11.293C22.9972 11.6835 22.9972 12.3167 22.6067 12.7072L20.0001 15.3138V19.0001C20.0001 19.5524 19.5524 20.0001 19.0001 20.0001H15.3138L12.7072 22.6067C12.3167 22.9972 11.6835 22.9972 11.293 22.6067L8.68637 20.0001H5.00008C4.4478 20.0001 4.00008 19.5524 4.00008 19.0001V15.3138L1.39348 12.7072C1.00295 12.3167 1.00295 11.6835 1.39348 11.293L4.00008 8.68637V5.00008C4.00008 4.4478 4.4478 4.00008 5.00008 4.00008H8.68637ZM6.00008 6.00008V9.5148L3.5148 12.0001L6.00008 14.4854V18.0001H9.5148L12.0001 20.4854L14.4854 18.0001H18.0001V14.4854L20.4854 12.0001L18.0001 9.5148V6.00008H14.4854L12.0001 3.5148L9.5148 6.00008H6.00008ZM12.0001 16.0001C9.79094 16.0001 8.00008 14.2092 8.00008 12.0001C8.00008 9.79094 9.79094 8.00008 12.0001 8.00008C14.2092 8.00008 16.0001 9.79094 16.0001 12.0001C16.0001 14.2092 14.2092 16.0001 12.0001 16.0001ZM12.0001 14.0001C13.1047 14.0001 14.0001 13.1047 14.0001 12.0001C14.0001 10.8955 13.1047 10.0001 12.0001 10.0001C10.8955 10.0001 10.0001 10.8955 10.0001 12.0001C10.0001 13.1047 10.8955 14.0001 12.0001 14.0001Z"},child:[]}]})(e)}const YS=()=>{const e=$l(),t=async n=>{n.preventDefault(),S_(ou),console.log("out"),e("/")};return k.jsxs("div",{className:"home",children:[k.jsxs("div",{className:"Menu",children:[k.jsx("img",{id:"logo1",src:Fl,alt:"logo"}),k.jsxs("ul",{children:[k.jsxs("li",{children:[k.jsx(KS,{}),"New Chat"]}),k.jsxs("li",{children:[k.jsx(qS,{}),"History"]}),k.jsxs("li",{children:[k.jsx(QS,{}),"Settings"]})]}),k.jsx("button",{className:"signOut",type:"submit",onSubmit:t,children:"Sign Out"})]}),"yy",k.jsxs("div",{className:"MainSearch",children:[k.jsx("input",{className:"search",type:"text",placeholder:"Enter article here.."}),k.jsx("button",{type:"submit",className:"upload",children:k.jsx(GS,{})})]})]})},JS=()=>k.jsx(k.Fragment,{children:k.jsx(C0,{children:k.jsxs(y0,{children:[k.jsx(gr,{exact:!0,path:"/SignUp.jsx",element:k.jsx(jS,{})}),k.jsx(gr,{exact:!0,path:"/",element:k.jsx(US,{})}),k.jsx(gr,{exact:!0,path:"/TermsAndConditions.jsx",element:k.jsx(FS,{})}),k.jsx(gr,{exact:!0,path:"/Home.jsx",element:k.jsx(YS,{})})]})})});function XS(){return k.jsx(k.Fragment,{children:k.jsx(JS,{})})}Fs.createRoot(document.getElementById("root")).render(k.jsx(st.StrictMode,{children:k.jsx(XS,{})}));
