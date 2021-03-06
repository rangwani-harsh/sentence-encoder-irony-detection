import React from 'react';
import HeatMap from './components/heatmap/HeatMap'
import PieChart from 'react-minimal-pie-chart';
import Collapsible from 'react-collapsible'

class ModelOutput extends React.Component {
  render() {

    const { outputs } = this.props;
    const { input } = this.props;
    // TODO: `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    var class_probabilities = outputs["class_probabilities"]
    
    var string_result_field = "Non-Ironic";
    var confidence = (1 - class_probabilities[1])*100;

    if (class_probabilities[1] >= 0.5) {
      string_result_field = "Ironic";
      confidence = class_probabilities[1] * 100;
    }

    // This is a 1D attention array, which we need to make into a 2D matrix to use with our heat
    // map component.
    var attention_data = outputs["attention"].map(x => [x]);
    console.log(attention_data)
    // This is a 2D attention matrix.
    //var matrix_attention_data = outputs['matrix_attention_data'];
    // Labels for our 2D attention matrix, and the rows in our 1D attention array.

    var row_labels = input["tweet"].split(/[ ,?]+/);
    //var row_labels = outputs['row_labels'];
    
    // This is how much horizontal space you'll get for the row labels.  Not great to have to
    // specify it like this, or with this name, but that's what we have right now.
    var xLabelWidth = "70px";

    return (
      <div className="model__content">

       {/*
         * TODO: This is where you display your output.  You can show whatever you want, however
         * you want.  We've got a few examples, of text-based output, and of visualizing model
         * internals using heat maps.
         */}
        
        <div className="form__field">
          <label>Model Prediction</label>
          <div className="model__content__summary" al>{ string_result_field } <br></br> With Confidence {confidence.toFixed(2) }%</div>
        </div>

        
        
        <div className="form__field">
          {/* We like using Collapsible to show model internals; you can keep this or change it. */}
          <Collapsible trigger="Model internals (beta)">
            <Collapsible trigger="1D attention">
                <HeatMap xLabels={['Words']} yLabels={row_labels} data={attention_data} xLabelWidth={xLabelWidth} />
            </Collapsible>
          </Collapsible>
        </div>
        
        <PieChart 
          animation = {true}
          radius = {30}
          data={[
          { title: 'Ironic', value: class_probabilities[0], color: '#E38627' },
          { title: 'Non-Ironic', value: class_probabilities[1], color: '#C13C37' },
          ]}

        />;
      </div>
    );
  }
}

export default ModelOutput;
