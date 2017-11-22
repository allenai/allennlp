import React from 'react';
import { Link } from 'react-router-dom';

/*******************************************************************************
  <Header /> Component
*******************************************************************************/

class Header extends React.Component {
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
        <header>
          <div className="header__content">
            <nav>
              <ul>
                {buildLink("machine-comprehension", "Machine Comprehension")}
                {buildLink("textual-entailment", "Textual Entailment")}
                {buildLink("semantic-role-labeling", "Semantic Role Labeling")}
                {buildLink("coreference-resolution", "Coreference Resolution")}
                {buildLink("named-entity-recognition", "Named Entity Recognition")}
              </ul>
            </nav>
            <h1 className="header__content__logo">
              <a href="http://www.allennlp.org/" target="_blank" rel="noopener noreferrer">
                <svg>
                  <use xlinkHref="#icon__allennlp-logo"></use>
                </svg>
                <span className="u-hidden">AllenNLP</span>
              </a>
            </h1>
          </div>
        </header>
      );
    }
  }

export default Header;
