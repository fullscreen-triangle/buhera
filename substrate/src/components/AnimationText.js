import { Fragment, useEffect, useState } from "react";
import TypeAnimation from "react-type-animation";

export const AnimationText1 = () => {
  return (
    <Fragment>
      <TypeAnimation
        cursor={true}
        sequence={[" Categorical Computing", 1500, " Trajectory Completion", 1500, " Zero-Cost Operations", 1500]}
        wrapper="span"
        repeat={Infinity}
      />
    </Fragment>
  );
};

const concepts = ["Trajectory Completion", "Categorical Addressing", "Zero-Cost Demons"];
export const RotateTextAnimation = () => {
  const [text, setText] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setText(text < concepts.length - 1 ? text + 1 : 0);
    }, 4000);
    return () => clearInterval(interval);
  });
  return (
    <Fragment>
      <span className="cd-headline rotate-1">
        <span className="blc">Powered by </span>
        <span className="cd-words-wrapper">
          {concepts.map((c, i) => (
            <b key={i} className={text === i ? "is-visible" : "is-hidden"}>
              {c}
            </b>
          ))}
        </span>
      </span>
    </Fragment>
  );
};
export const ZoomTextAnimation = () => {
  const [text, setText] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => {
      setText(text < concepts.length - 1 ? text + 1 : 0);
    }, 5000);
    return () => clearInterval(interval);
  });
  return (
    <Fragment>
      <span className="cd-headline zoom">
        <span className="blc">Powered by </span>
        <span className="cd-words-wrapper">
          {concepts.map((c, i) => (
            <b key={i} className={text === i ? "is-visible" : "is-hidden"}>
              {c}
            </b>
          ))}
        </span>
      </span>
    </Fragment>
  );
};
export const LoadingTextAnimation = () => {
  const [text, setText] = useState(0);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const interval = setInterval(() => {
      setText(text < concepts.length - 1 ? text + 1 : 0);
      setLoading(!loading);
    }, 5000);
    if (loading == false) {
      setTimeout(() => {
        setLoading(true);
      }, 100);
    }
    return () => clearInterval(interval);
  });
  return (
    <Fragment>
      <span className="cd-headline loading-bar">
        <span className="blc">Powered by </span>
        <span className={`cd-words-wrapper ${loading ? "is-loading" : ""}`}>
          {concepts.map((c, i) => (
            <b key={i} className={text === i ? "is-visible" : "is-hidden"}>
              {c}
            </b>
          ))}
        </span>
      </span>
    </Fragment>
  );
};
