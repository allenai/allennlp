import React from 'react';
import { Link } from 'react-router-dom';

/*******************************************************************************
  <Header /> Component
*******************************************************************************/

class Menu extends React.Component {
    render() {
      const { selectedModel, clearData } = this.props;

      const buildLink = (thisModel, label) => {
        return (
          <li>
            <span className={`nav__link ${selectedModel === thisModel ? "nav__link--selected" : ""}`}>
              <Link to={"/" + thisModel} onClick={clearData}>
                <span>{label}</span>
              </Link>
            </span>
          </li>
        )
      }

      return (
        <div className="menu">
          <div className="menu__content">
            <h1 className="menu__content__logo">
              <a href="http://www.allennlp.org/" target="_blank" rel="noopener noreferrer">
                <svg>
                  <use xlinkHref="#icon__allennlp-logo"></use>
                </svg>
                <span className="u-hidden">AllenNLP</span>
              </a>
            </h1>
            <nav>
              <ul>
                {buildLink("machine-comprehension", "Machine Comprehension")}
                {buildLink("textual-entailment", "Textual Entailment")}
                {buildLink("semantic-role-labeling", "Semantic Role Labeling")}
                {buildLink("coreference-resolution", "Coreference Resolution")}
                {buildLink("named-entity-recognition", "Named Entity Recognition")}
                {buildLink("constituency-parsing", "Constituency Parsing")}
                {buildLink("user-models", "Your model here!")}
              </ul>
            </nav>
          </div>
        </div>
      );
    }
  }

export default Menu;
