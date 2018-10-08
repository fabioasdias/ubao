import React, { Component } from 'react';
import './App.css';

function getData(url,action){
  fetch(url)
  .then((response) => {
    if (response.status >= 400) {
      throw new Error("Bad response from server");
    }
    return response.json();
  })
  .then((data) => {
    action(data);
  });

}
let HN=8;
let VN=5;// number of H and V slots
let MaxH=950;
let MaxW=1700;

function strSlot(ls,ts){
  let ret='('+String(ls)+','+String(ts)+')';
  //console.log(ls,ts,ret);
  return(ret);
}

class App extends Component {
  constructor(props){
    super(props);
    this.state={imagelist:[], nextImage:0, z:1, lastUsed:{}};
  }
  componentDidMount() {
    // getData('master.json',(master)=>{
      // getData(master[0].fname, (data)=>{
        getData('Panel_78.json', (data)=>{
          for (let i=0;i<data.length;i++){
            data[i].visibility='hidden';
            data[i].z=1;
            data[i].left_slot=Math.floor(HN*data[i].left);
            data[i].top_slot=Math.floor(VN*data[i].top);
          }
          this.setState({imagelist:data});    
      });
    // });

    if(!this.timerId){     
      this.timerId = setInterval(()=>{
        if (this.state.imagelist.length>0)
        {
          let temp=this.state.imagelist.slice();
          let n=this.state.nextImage;
          let nextSlot=strSlot(temp[n].left_slot,temp[n].top_slot);
          if (this.state.lastUsed.hasOwnProperty(nextSlot)){
            temp[this.state.lastUsed[nextSlot]].visibility='hidden';
          }
          let last={...this.state.lastUsed};
          last[nextSlot]=n;
          temp[n].visibility='visible';
          temp[n].z=this.state.z+1;
          // console.log(n,nextSlot)
          this.setState({imagelist:temp.slice(),
                          lastUsed:last,
                          nextImage:(Math.floor(Math.random()*temp.length)),
                          z:(this.state.z+1)});
        }    
      }, 10);
    }
  }
  componentWillUnmount() {
    clearInterval(this.interval);
  }
  render() {
    if ((this.state!==undefined) && (this.state.imagelist!==undefined) && (this.state.imagelist.length>0)) {
      return (
        <div>
          {this.state.imagelist.map((d) => {
            return(<img 
                      className="image" 
                      key={d.fname} 
                      style={{left:MaxW*(d.left_slot/HN),
                              top:MaxH*(d.top_slot/VN), 
                              visibility:d.visibility, 
                              width:'auto',
                              height:'auto',
                              maxHeight:'175px', 
                              maxWidth:'175px', 
                              zIndex:d.z}} 
                      src={'./img/'+d.fname} 
                      alt={d.fname} 
                      />)
          })}
        </div>)
      }
    else{return(null);}
  }
}
export default App;