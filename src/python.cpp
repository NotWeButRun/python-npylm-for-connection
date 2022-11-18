#include "python/dataset.h"
#include "python/dictionary.h"
#include "python/model.h"
#include "python/trainer.h"

using namespace npylm;
using boost::python::arg;

// enable copy /////////////////////////////////////

#define PYTHON_ERROR(TYPE, REASON) \
{ \
    PyErr_SetString(TYPE, REASON); \
    throw boost::python::error_already_set(); \
}

template<class T>
inline PyObject * managingPyObject(T *p)
{
    return typename boost::python::manage_new_object::apply<T *>::type()(p);
}

template<class Copyable>
boost::python::object generic__copy__(boost::python::object copyable)
{
    Copyable *newCopyable(new Copyable(boost::python::extract<const Copyable&>(copyable)));
    boost::python::object result(boost::python::detail::new_reference(managingPyObject(newCopyable)));

    boost::python::extract<boost::python::dict>(result.attr("__dict__"))().update(
        copyable.attr("__dict__"));

    return result;
}

template<class Copyable>
boost::python::object generic__deepcopy__(boost::python::object copyable, boost::python::dict memo)
{
    boost::python::object copyMod = boost::python::import("copy");
    boost::python::object deepcopy = copyMod.attr("deepcopy");

    Copyable *newCopyable(new Copyable(boost::python::extract<const Copyable&>(copyable)));
    boost::python::object result(boost::python::detail::new_reference(managingPyObject(newCopyable)));

    // HACK: copyableId shall be the same as the result of id(copyable) in Python -
    uintptr_t copyableId = (uintptr_t)(copyable.ptr());
    memo[copyableId] = result;

    boost::python::extract<boost::python::dict>(result.attr("__dict__"))().update(
        deepcopy(boost::python::extract<boost::python::dict>(copyable.attr("__dict__"))(),memo)
    );

    return result;
}

////////////////////////////////////////////////////



BOOST_PYTHON_MODULE(npylm)
{
    boost::python::class_<Dictionary>("dictionary")
        .def("save", &Dictionary::save)
        .def("load", &Dictionary::load)
        .def("__copy__", &generic__copy__<Dictionary>)
        .def("__deepcopy__", &generic__deepcopy__<Dictionary>);
        // .def(init< const Dictionary & >())

    boost::python::class_<Corpus>("corpus")
        .def("add_textfile", &Corpus::add_textfile)
        .def("add_true_segmentation", &Corpus::python_add_true_segmentation)
        .def("add_sentence", &Corpus::add_sentence)
        .def("__copy__", &generic__copy__< Corpus >)
        .def("__deepcopy__", &generic__deepcopy__< Corpus >);
        // .def(init< const Corpus & >());


    boost::python::class_<Dataset>("dataset", boost::python::init<Corpus*, double, int>())
        .def("get_max_sentence_length", &Dataset::get_max_sentence_length)
        .def("detect_hash_collision", &Dataset::detect_hash_collision)
        .def("get_num_sentences_train", &Dataset::get_num_sentences_train)
        .def("get_num_sentences_dev", &Dataset::get_num_sentences_dev)
        .def("get_num_sentences_supervised", &Dataset::get_num_sentences_supervised)
        .def("get_dict", &Dataset::get_dict_obj, boost::python::return_internal_reference<>())
        .def("__copy__", &generic__copy__< Dataset >)
        .def("__deepcopy__", &generic__deepcopy__< Dataset >);
        // .def(init< const Dataset & >());


    boost::python::class_<Trainer>("trainer", boost::python::init<Dataset*, Model*, bool>((arg("dataset"), arg("model"), arg("always_accept_new_segmentation") = true)))
        .def("print_segmentation_train", &Trainer::print_segmentation_train)
        .def("print_segmentation_dev", &Trainer::print_segmentation_dev)
        .def("sample_hpylm_vpylm_hyperparameters", &Trainer::sample_hpylm_vpylm_hyperparameters)
        .def("sample_lambda", &Trainer::sample_lambda)
        .def("update_p_k_given_vpylm", &Trainer::update_p_k_given_vpylm)
        .def("compute_perplexity_train", &Trainer::compute_perplexity_train)
        .def("compute_perplexity_dev", &Trainer::compute_perplexity_dev)
        .def("gibbs", &Trainer::gibbs)
        .def("__copy__", &generic__copy__< Trainer >)
        .def("__deepcopy__", &generic__deepcopy__< Trainer >);
        // .def(init< const Trainer & >());


    boost::python::class_<Model>("model", boost::python::init<Dataset*, int>())
        .def(boost::python::init<std::string>())
        .def("set_initial_lambda_a", &Model::set_initial_lambda_a)
        .def("set_initial_lambda_b", &Model::set_initial_lambda_b)
        .def("set_vpylm_beta_stop", &Model::set_vpylm_beta_stop)
        .def("set_vpylm_beta_pass", &Model::set_vpylm_beta_pass)
        .def("get_lambda", &Model::python_get_lambda)
        .def("parse", &Model::python_parse)
        .def("save", &Model::save)
        .def("load", &Model::load)
        .def("__copy__", &generic__copy__< Model >)
        .def("__deepcopy__", &generic__deepcopy__< Model >);
        // .def(init< const Model & >());
}