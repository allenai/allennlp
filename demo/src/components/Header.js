import React from 'react';

/*******************************************************************************
  <Header /> Component
*******************************************************************************/

class Header extends React.Component {
    render() {
      const { selectedModel, changeModel } = this.props;

      const buildLink = (thisModel, label) => {
        return (
          <li>
            <a href="#" className={`nav__link ${selectedModel === thisModel ? "nav__link--selected" : ""}`} onClick={() => { changeModel(thisModel) }}>
              <span>{label}</span>
            </a>
          </li>
        )
      }

      return (
        <header>
          <div className="header__content">
            <nav>
              <ul>
                {buildLink("srl", "SRL Model")}
                {buildLink("mc", "MC Model")}
                {buildLink("te", "TE Model")}
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
