import React, { Component } from 'react';
import './App.css';

class App extends Component {
  constructor(props){
    super(props);
    this.state={imagelist:[], nextImage:0, z:1};
  }
  componentDidMount() {
    let url='imglist.json';
    fetch(url)
    .then((response) => {
      if (response.status >= 400) {
        throw new Error("Bad response from server");
      }
      return response.json();
    })
    .then((data) => {
      for (let i=0;i<data.length;i++){
        data[i].visibility='hidden';
        data[i].z=1;
      }
      this.setState({imagelist:data});
    });

    if(!this.timerId){     
      this.timerId = setInterval(()=>{
        if (this.state.imagelist.length>0)
        {
          let temp=this.state.imagelist.slice();
          let n=this.state.nextImage;
          // if (n>=temp.length){
          //   n=0;
          // }
          temp[n].visibility='visible';
          temp[n].z=this.state.z+1;
          this.setState({imagelist:temp.slice(),nextImage:(Math.floor(Math.random()*temp.length)),z:(this.state.z+1)});
        }    
      }, 50);
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
            return(<img className="image" key={d.fname} style={{left:1600*(d.left),top:800*(d.top), visibility:d.visibility, zIndex:d.z}} src={'./img/'+d.fname} alt={d.fname} />)
          })}
        </div>)
      }
    else{return(null);}
  }
}
export default App;